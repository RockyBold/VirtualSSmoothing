import torch
import torch.nn.functional as F
import numpy as np

def kl_loss_from_prob(student_prob, teacher_prob):
    batch_size = student_prob.size()[0]
    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    log_student_prob = torch.log(student_prob)
    kl_loss = (1.0 / batch_size) * criterion_kl(log_student_prob, teacher_prob)
    return kl_loss


# def virtual_self_distillion_loss(student_logits, num_classes, temperature, alpha, y_true):
#     # batch_size = student_logits.size()[0]
#     student_prob = F.softmax(student_logits, dim=1)
#
#     teacher_logits = torch.zeros_like(student_logits)
#     teacher_logits[:, num_classes:] = student_logits[:, :num_classes]
#     indexes = [i for i in range(len(student_logits))]
#     teacher_logits[indexes, num_classes + y_true] = 0.0
#     teacher_logits[indexes, y_true] = student_logits[indexes, y_true]
#     teacher_prob = F.softmax(teacher_logits, dim=1)
#
#     # KL divergence * (temperature^2)
#     kl_loss = temperature * temperature * kl_loss_from_prob(student_prob / temperature, teacher_prob / temperature)
#     xent = torch.nn.CrossEntropyLoss()
#     ce_loss = xent(student_logits, y_true)
#     loss = alpha * kl_loss + (1 - alpha) * ce_loss
#     return loss


def rob_distillion_loss(nat_student_logits, adv_student_logits, nat_teacher_logits, temperature, alpha, y_true):
    KL_loss = torch.nn.KLDivLoss()
    XENT_loss = torch.nn.CrossEntropyLoss()
    kl_loss = KL_loss(F.log_softmax(adv_student_logits / temperature, dim=1),
                      F.softmax(nat_teacher_logits / temperature, dim=1))
    ce_loss = XENT_loss(nat_student_logits, y_true)
    loss = temperature * temperature * alpha * kl_loss + (1 - alpha) * ce_loss
    return loss


def virtual_self_distillion_loss(student_logits, num_classes, temperature, alpha, y_true):
    # batch_size = student_logits.size()[0]
    student_prob = F.softmax(student_logits, dim=1)

    teacher_prob = torch.zeros_like(student_logits)
    teacher_prob[:, num_classes:] = student_prob[:, :num_classes]
    indexes = [i for i in range(len(student_logits))]
    teacher_prob[indexes, num_classes + y_true] = 0.0
    teacher_prob[indexes, y_true] = student_prob[indexes, y_true]

    # KL divergence * (temperature^2)
    kl_loss = temperature * temperature * kl_loss_from_prob(student_prob / temperature, teacher_prob / temperature)
    xent = torch.nn.CrossEntropyLoss()
    ce_loss = xent(student_logits, y_true)
    loss = alpha * kl_loss + (1 - alpha) * ce_loss
    return loss


def cross_entropy_soft_target(logit, y_soft):
    batch_size = logit.size()[0]
    log_prob = F.log_softmax(logit, dim=1)
    loss = -torch.sum(log_prob * y_soft) / batch_size
    return loss


def eval_in_v_from_data(model, x, y, batch_size, num_real_classes):
    model.eval()
    num_examples = len(x)
    in_corr = 0
    v_corr = 0
    in_v_corr = 0
    total = 0

    for idx in range(0, len(x), batch_size):
        st_idx = idx
        end_idx = min(idx + batch_size, num_examples)
        batch_x = x[st_idx:end_idx]
        batch_y = y[st_idx:end_idx]
        # compute output
        with torch.no_grad():
            logits = model(batch_x)
            num_v_classes = logits.size(1) - num_real_classes
            assert num_v_classes > 0

        _, in_pred = torch.max(logits[:, :num_real_classes], dim=1)
        in_corr_idx = in_pred == batch_y
        in_corr += in_corr_idx.sum()
        if num_v_classes == num_real_classes:
            _, v_pred = torch.max(logits[:, num_real_classes:], dim=1)
            _, in_v_pred = torch.max(logits[:, :num_real_classes] + logits[:, num_real_classes:], dim=1)
            v_corr_idx = v_pred == batch_y
            in_v_corr_idx = in_v_pred == batch_y
            v_corr += v_corr_idx.sum()
            in_v_corr += in_v_corr_idx.sum()
        total += batch_y.size(0)
    in_acc = float(in_corr) / total
    v_acc = float(v_corr) / total
    in_v_acc = float(in_v_corr) / total
    return in_acc, v_acc, in_v_acc


def eval_in_v(model, test_loader, num_real_classes, device=torch.device("cuda")):
    model.eval()
    in_corr = 0
    v_corr = 0
    in_v_corr = 0
    total = 0

    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_y = batch_y.to(device)
        batch_x = batch_x.to(device)

        with torch.no_grad():
            logits = model(batch_x)
            num_v_classes = logits.size(1) - num_real_classes
            assert num_v_classes > 0

        _, in_pred = torch.max(logits[:, :num_real_classes], dim=1)
        in_corr_idx = in_pred == batch_y
        in_corr += in_corr_idx.sum()
        if num_v_classes == num_real_classes:
            _, v_pred = torch.max(logits[:, num_real_classes:], dim=1)
            _, in_v_pred = torch.max(logits[:, :num_real_classes] + logits[:, num_real_classes:], dim=1)
            v_corr_idx = v_pred == batch_y
            in_v_corr_idx = in_v_pred == batch_y
            v_corr += v_corr_idx.sum()
            in_v_corr += in_v_corr_idx.sum()
        total += batch_y.size(0)
    in_acc = float(in_corr) / total
    v_acc = float(v_corr) / total
    in_v_acc = float(in_v_corr) / total
    return in_acc, v_acc, in_v_acc


def eval(model, test_loader, num_real_classes, device=torch.device("cuda")):
    model.eval()
    correct = 0
    total = 0
    num_max_in_v = 0
    num_corr_max_in_v = 0

    corr_conf = []
    corr_v_conf = []

    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_y = batch_y.to(device)
        batch_x = batch_x.to(device)
        u_idx = torch.arange(0, len(batch_y))

        with torch.no_grad():
            logits = model(batch_x)
            conf = F.softmax(logits, dim=1)

        _, pred = torch.max(logits[:, :num_real_classes], dim=1)
        corr_idx = pred == batch_y
        correct += corr_idx.sum()
        total += batch_y.size(0)

        _, whole_pred = torch.max(logits, dim=1)
        max_in_v_indices = whole_pred >= num_real_classes
        num_max_in_v += max_in_v_indices.sum().item()

        num_corr_max_in_v += (torch.logical_and(corr_idx, max_in_v_indices)).sum().item()

        corr_conf.append(conf[u_idx, batch_y][corr_idx])
        corr_v_conf.append(conf[corr_idx, num_real_classes:])
    corr_conf = torch.cat(corr_conf, dim=0)
    corr_v_conf = torch.cat(corr_v_conf, dim=0)
    acc = (float(correct) / total) * 100
    return acc, num_max_in_v, num_corr_max_in_v, corr_conf, corr_v_conf


def eval_from_data(model, x, y, batch_size, num_real_classes, align_x=None):
    model.eval()
    num_examples = len(x)
    correct = 0
    total = 0
    num_max_in_v = 0
    num_corr_max_in_v = 0
    
    corr_conf = []
    corr_v_conf = []
    
    for idx in range(0, len(x), batch_size):
        st_idx = idx
        end_idx = min(idx + batch_size, num_examples)
        batch_x = x[st_idx:end_idx]
        batch_y = y[st_idx:end_idx]
        idx_u = torch.arange(0, len(batch_y))
        # compute output
        with torch.no_grad():
            logits = model(batch_x)
            conf = F.softmax(logits, dim=1)

        _, pred = torch.max(logits[:, :num_real_classes], dim=1)
        corr_idx = pred == batch_y
        correct += corr_idx.sum()
        total += batch_y.size(0)

        _, whole_pred = torch.max(logits, dim=1)
        max_in_v_indices = whole_pred >= num_real_classes
        num_max_in_v += max_in_v_indices.sum().item()

        num_corr_max_in_v += (torch.logical_and(corr_idx, max_in_v_indices)).sum().item()

        align_idx = None
        if align_x is not None:
            batch_a_x = align_x[st_idx:end_idx]
            with torch.no_grad():
                a_pred = model(batch_a_x)[:, :num_real_classes].max(dim=1)[1]
            align_idx = a_pred == batch_y
        if align_idx is not None:
            corr_conf.append(conf[idx_u, batch_y][align_idx])
            corr_v_conf.append(conf[align_idx, num_real_classes:])
        else:
            corr_conf.append(conf[idx_u, batch_y][corr_idx])
            corr_v_conf.append(conf[corr_idx, num_real_classes:])
    corr_conf = torch.cat(corr_conf, dim=0)
    corr_v_conf = torch.cat(corr_v_conf, dim=0)

    acc = (float(correct) / total) * 100
    return acc, num_max_in_v, num_corr_max_in_v, corr_conf, corr_v_conf


def mixup_data(x, y, alpha=0.8, num_classes=10):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    soft_y = F.one_hot(y, num_classes=num_classes).to(y.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * soft_y + (1 - lam) * soft_y[index, :]

    return mixed_x, mixed_y, lam


def mixup_data_prob(x, y, prob=0.8, num_classes=10):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    soft_y = F.one_hot(y, num_classes=num_classes).to(y.device)

    mixed_x = prob * x + (1 - prob) * x[index, :]
    mixed_y = prob * soft_y + (1 - prob) * soft_y[index, :]

    return mixed_x, mixed_y

if __name__ == '__main__':
    student_logits = torch.tensor([[0.9, -0.1, 0.1, float('-inf'), 0, 0, 0, 0], [0.9, -0.1, 0.1, 0, 0, 0, 0, 0]])
    num_classes = 4
    y_true = torch.tensor([0, 0])
    virtual_self_distillion_loss(student_logits, num_classes, 3, 0.0, y_true)
#     student_logits = torch.rand((4, 10))
#     teacher_logits = torch.rand((4, 10))
#
#     criterion_kl = torch.nn.KLDivLoss(size_average=False)
#     kl_loss1 = (1.0 / 4) * criterion_kl(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1))
#     print(kl_loss1)
#
#     kl_loss2 = kl_loss_from_prob(F.softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1))
#     print(kl_loss2)
