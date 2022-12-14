import visualkeras

class VisualKeras:
    """
    This Class will be used for high level visualization of keras models using VisualKeras.
    """
    def visualize_model_using_vk(self, model):
        visualkeras.layered_view(model, legend=True).show()
