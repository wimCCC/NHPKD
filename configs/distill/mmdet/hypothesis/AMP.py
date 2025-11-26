_base_ = ['./LAAP.py']

model = dict(
    distiller=dict(
        distill_losses=dict(
            loss_pkd_fpn0=dict(loss_weight=2, type='PAMLoss'),
            loss_pkd_fpn1=dict(loss_weight=2, type='PAMLoss'),
            loss_pkd_fpn2=dict(loss_weight=2, type='PAMLoss'),
            loss_pkd_fpn3=dict(loss_weight=2, type='PAMLoss'))))

work_dir = '/home4/caiwm/hypothesis/PAM'
