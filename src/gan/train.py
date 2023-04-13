import torch
from src.utils.checkpoint import checkpoint_gan, checkpoint_image
from src.utils import seed_worker
from tqdm import tqdm
import math
from src.utils import MetricsLogger, group_images
import matplotlib.pyplot as plt


def loss_terms_to_str(loss_items):
    result = ''
    for key, value in loss_items.items():
        result += '%s: %.4f ' % (key, value)

    return result


def evaluate(G, fid_metrics, stats_logger, batch_size, test_noise, device, c_out_hist):
    # Compute evaluation metrics on fixed noise (Z) set
    training = G.training
    G.eval()

    start_idx = 0
    num_batches = math.ceil(test_noise.size(0) / batch_size)

    for _ in tqdm(range(num_batches), desc="Evaluating"):
        real_size = min(batch_size, test_noise.size(0) - start_idx)

        batch_z = test_noise[start_idx:start_idx + real_size]

        with torch.no_grad():
            batch_gen = G(batch_z.to(device))

        for metric_name, metric in fid_metrics.items():
            metric.update(batch_gen, (start_idx, real_size))

        if c_out_hist is not None:
            c_out_hist.update(batch_gen, (start_idx, real_size))

        start_idx += batch_z.size(0)

    for metric_name, metric in fid_metrics.items():
        result = metric.finalize()
        stats_logger.update_epoch_metric(metric_name, result, prnt=True)
        metric.reset()

    if c_out_hist is not None:
        c_out_hist.plot()
        stats_logger.log_plot('histogram') # TODO verify if we still have this error
        c_out_hist.reset()
        plt.clf()

    if training:
        G.train()


def train_disc(G, D, d_opt, d_crit, real_data,
               batch_size, train_metrics, device):
    D.zero_grad()

    # Real data batch
    real_data = real_data.to(device)
    d_output_real = D(real_data)

    # Fake data batch
    noise = torch.randn(batch_size, G.z_dim, device=device)
    with torch.no_grad():
        fake_data = G(noise)

    d_output_fake = D(fake_data.detach())

    # Compute loss, gradients and update net
    d_loss, d_loss_terms = d_crit(real_data, fake_data,
                                  d_output_real, d_output_fake, device)
    d_loss.backward()
    d_opt.step()

    for loss_term_name, loss_term_value in d_loss_terms.items():
        train_metrics.update_it_metric(loss_term_name, loss_term_value)

    train_metrics.update_it_metric('D_loss', d_loss.item())

    return d_loss, d_loss_terms


def train_gen(update_fn, G, D, g_opt,
              batch_size, train_metrics, device):
    noise = torch.randn(batch_size, G.z_dim, device=device)

    g_loss, g_loss_terms = update_fn(G, D, g_opt, noise, device)

    for loss_term_name, loss_term_value in g_loss_terms.items():
        train_metrics.update_it_metric(loss_term_name, loss_term_value)

    train_metrics.update_it_metric('G_loss', g_loss.item())

    return g_loss, g_loss_terms


def train(config, dataset, device, n_epochs, batch_size, G, g_opt, g_updater, D,
          d_opt, d_crit, test_noise, fid_metrics, n_disc_iters,
          early_stop=None,  # Tuple of (key, crit)
          start_early_stop_when=None,  # Tuple of (key, crit):
          checkpoint_dir=None, checkpoint_every=10, fixed_noise=None, c_out_hist=None,
          classifier=None):

    fixed_noise = torch.randn(
        64, G.z_dim, device=device) if fixed_noise is None else fixed_noise

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=config["num-workers"], worker_init_fn=seed_worker)

    train_metrics = MetricsLogger(prefix='train')
    eval_metrics = MetricsLogger(prefix='eval')

    train_state = {
        'epoch': 0,
        'early_stop_tracker': 0,
        'best_epoch': 0,
        'best_epoch_metric': float('inf'),
    }

    early_stop_state = 2
    if early_stop[1] is not None:
        early_stop_key, early_stop_crit = early_stop
        early_stop_state = 1
        if start_early_stop_when is not None:
            train_state['pre_early_stop_tracker'] = 0,
            train_state['pre_early_stop_metric'] = float('inf')
            pre_early_stop_key, pre_early_stop_crit = start_early_stop_when
            early_stop_state = 0

    train_metrics.add('G_loss', iteration_metric=True)
    train_metrics.add('D_loss', iteration_metric=True)

    for loss_term in g_updater.get_loss_terms():
        train_metrics.add(loss_term, iteration_metric=True)

    for loss_term in d_crit.get_loss_terms():
        train_metrics.add(loss_term, iteration_metric=True)

    for metric_name in fid_metrics.keys():
        eval_metrics.add(metric_name)

    eval_metrics.add_media_metric('samples')
    eval_metrics.add_media_metric('histogram')

    with torch.no_grad():
        G.eval()
        fake = G(fixed_noise).detach().cpu()
        G.train()

    latest_cp = checkpoint_gan(
        G, D, g_opt, d_opt, {}, {}, config, epoch=0, output_dir=checkpoint_dir)
    best_cp = latest_cp

    img = group_images(fake, classifier=classifier, device=device)
    checkpoint_image(img, 0, output_dir=checkpoint_dir)

    G.train()
    D.train()

    g_iters_per_epoch = int(math.floor(len(dataloader) / n_disc_iters))
    iters_per_epoch = g_iters_per_epoch * n_disc_iters

    log_every_g_iter = 50

    print("Training...")
    for epoch in range(1, n_epochs+1):
        data_iter = iter(dataloader)
        curr_g_iter = 0

        for i in range(1, iters_per_epoch+1):
            data, _ = next(data_iter)
            real_data = data.to(device)

            ###
            # Update Discriminator
            ###
            d_loss, d_loss_terms = train_disc(
                G, D, d_opt, d_crit, real_data, batch_size, train_metrics, device)

            ###
            # Update Generator
            # - update every 'n_disc_iterators' consecutive D updates
            ###
            if i % n_disc_iters == 0:
                curr_g_iter += 1

                g_loss, g_loss_terms = train_gen(g_updater,
                    G, D, g_opt, batch_size, train_metrics, device)

                ###
                # Log stats
                ###
                if curr_g_iter % log_every_g_iter == 0 or \
                        curr_g_iter == g_iters_per_epoch:
                    print('[%d/%d][%d/%d]\tG loss: %.4f %s; D loss: %.4f %s'
                          % (epoch, n_epochs, curr_g_iter, g_iters_per_epoch, g_loss.item(), loss_terms_to_str(g_loss_terms), d_loss.item(),
                             loss_terms_to_str(d_loss_terms)))

        ###
        # Sample images
        ###
        with torch.no_grad():
            G.eval()
            fake = G(fixed_noise).detach().cpu()
            G.train()

        img = group_images(fake, classifier=classifier, device=device)
        checkpoint_image(img, epoch, output_dir=checkpoint_dir)
        eval_metrics.log_image('samples', img)

        ###
        # Evaluate after epoch
        ###
        train_state['epoch'] += 1

        train_metrics.finalize_epoch()

        evaluate(G, fid_metrics, eval_metrics, batch_size,
                 test_noise, device, c_out_hist)

        eval_metrics.finalize_epoch()

        ###
        # Checkpoint GAN
        ###
        if epoch == n_epochs or epoch % checkpoint_every == 0:
            latest_cp = checkpoint_gan(
                G, D, g_opt, d_opt, train_state, {"eval": eval_metrics.stats, "train": train_metrics.stats}, config, epoch=epoch, output_dir=checkpoint_dir)

        ###
        # Test for early stopping
        ###
        if early_stop_state == 2:
            best_cp = latest_cp
        elif early_stop_state == 0:
            # Pre early stop phase
            if eval_metrics.stats[pre_early_stop_key][-1] \
                    < train_state['pre_early_stop_metric']:
                train_state['pre_early_stop_tracker'] = 0
                train_state['pre_early_stop_metric'] = \
                    eval_metrics.stats[pre_early_stop_key][-1]
            else:
                train_state['pre_early_stop_tracker'] += 1
                print(
                    " > Pre-early stop tracker {}/{}".format(train_state['pre_early_stop_tracker'], pre_early_stop_crit))
                if train_state['pre_early_stop_tracker'] \
                        == pre_early_stop_crit:
                    early_stop_state = 1

            best_cp = latest_cp
        else:
            # Early stop phase
            if eval_metrics.stats[early_stop_key][-1] < train_state['best_epoch_metric']:
                train_state['early_stop_tracker'] = 0
                train_state['best_epoch'] = epoch + 1
                train_state['best_epoch_metric'] = eval_metrics.stats[
                    early_stop_key][-1]
                best_cp = latest_cp
            else:
                train_state['early_stop_tracker'] += 1
                print(
                    " > Early stop tracker {}/{}".format(train_state['early_stop_tracker'], early_stop_crit))
                if train_state['early_stop_tracker'] == early_stop_crit:
                    break

    return train_state, best_cp, train_metrics, eval_metrics
