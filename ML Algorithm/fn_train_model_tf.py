def train_model(model, opt, loss, met, set, labels, bs, e, vset, vlabels, save_name):
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=met
    )
    history1 = model.fit(
        set,
        labels,
        batch_size=bs,
        epochs=e,
        validation_data=(vset, vlabels)
    )
    model.save(save_name)