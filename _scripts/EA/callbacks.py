class Callback(object):
    """Abstract base class used to build new callbacks. Based on the
    Callback class from Keras.
    This class is used to store functions that are called in certain points of the evolutionary algorithm.
    """

    def __init__(self):
        self.setup = None
        self.model = None

    def set_setup(self, setup):
        self.setup = setup

    def set_model(self, model):
        self.model = model

    def on_initialize_begin(self, logs=None):
        pass

    def on_initialize_end(self, logs=None):
        pass

    def on_generation_begin(self, generation, logs=None):
        pass

    def on_generation_end(self, generation, logs=None):
        pass

    def on_evaluation_begin(self, population, logs=None):
        pass

    def on_evaluation_end(self, population, logs=None):
        pass


class CallbackList(object):
    """Container abstracting a list of callbacks, based on
    the Keras implementation of Callbacks.
    # Arguments
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    """

    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def set_setup(self, params):
        for callback in self.callbacks:
            callback.set_setup(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_initialize_begin(self, logs=None):

        logs = logs or {}
        for callback in self.callbacks:
            callback.on_initialize_begin(logs)

    def on_initialize_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_initialize_end(logs)

    def on_generation_begin(self, generation, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_generation_begin(generation, logs)

    def on_generation_end(self, generation, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_generation_end(generation, logs)

    def on_evaluation_begin(self, population, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_evaluation_begin(population, logs)

    def on_evaluation_end(self, population, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_evaluation_end(population, logs)

    def __iter__(self):
        return iter(self.callbacks)
