prompt = f"""You are a image prompt generator generate 5 prompts based strictly on this template with the objects seen clearly"""
# Define your template
template = """
        a {object_scene} with {object}, {object} and an {object}, {object} ordered,
        a {object}, a {object} and an {object} on a {object} {object_scene} precisely,
        an {object} on a {object}, a {object}  and a {object} geometric separate, {object_scene}
        
        object = "knife", "tomatoe", "glass bottle","spoon", "mug", "cup", "bowl",
        "egg", "orange","apple", "porcelain", "open pot", "pan","plate"
        object_scene = "tabletop"
        """

