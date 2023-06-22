import yaml
def read_config(learning):
    clinical_config = yaml.safe_load(open("config/clinical/general.yml", "r"))
    if learning:
        learning_config = yaml.safe_load(open("config/clinical/learning.yml", "r"))
        
        # Append cognitive batteries needed for test-based digas to relevant assessments 
        cog_batteries = clinical_config["cog batteries"]
        clinical_config["relevant assessments"] += list(cog_batteries)

        # Rewrite all keys in clnical with values in learning
        clinical_config.update(learning_config)

    print("DEBUG", clinical_config)
    return clinical_config