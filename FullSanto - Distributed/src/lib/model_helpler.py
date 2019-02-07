def load_best_model_weight(model):
    return model.load(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)

def save_as_best_model(model):
    return model.save(model.config.resource.model_best_config_path, model.config.resource.model_best_weight_path)

def reload_best_model_weight_if_changed(model):
    print("start reload the best model if changed")
    digest = model.fetch_digest(model.config.resource.model_best_weight_path)
    if digest != model.digest:
        return load_best_model_weight(model)

    print("the best model is not changed")
    return False
