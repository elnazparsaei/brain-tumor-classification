class_mappings = {'Glioma': 0, 'Meninigioma': 1, 'Notumor': 2, 'Pituitary': 3}
inv_class_mapping = {v: k for k, v in class_mappings.items()}
class_names = list(class_mappings.keys())