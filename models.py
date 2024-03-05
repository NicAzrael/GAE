import tensorflow as tf
from tensorflow.keras import Model
from config import cfg

INPUT_SHAPE = [cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.NUM_CHANNELS]

class target_model():
    def __init__(self, model_path=cfg.MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model_logits = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)

    def predict_logits(self, imgs):
        logits = self.model_logits(imgs)
        probs = tf.nn.sigmoid(logits)
        return logits, probs

    def predict_softmax(self, imgs):
        return self.model(imgs)

def Generator():

    inp = tf.keras.layers.Input(shape=INPUT_SHAPE, name='input_image')
    x = convolutional(inp, (3, 3, 3, 32), downsample=True, bn=False)  
    x = convolutional(x, (3, 3, 32, 64), downsample=True)    
    x = convolutional(x, (3, 3, 64, 128), downsample=True)  
    x = convolutional(x, (3, 3, 128, 256), downsample=True)  
    x = convolutional(x, (3, 3, 256, 512), downsample=True) 

    for _ in range(5):
        x = residual_block(x, 512, 256, 512)

    x = upsample(x, (3, 3, 512, 256))  
    x = upsample(x, (3, 3, 256, 128))  
    x = upsample(x, (3, 3, 128, 64))   
    x = upsample(x, (3, 3, 64, 32))    
    x = upsample(x, (3, 3, 32, 3), bn=False)    

    return tf.keras.Model(inputs=inp, outputs=x)


def Discriminator():
    inp = tf.keras.layers.Input(shape=INPUT_SHAPE, name='input_image')
    x = convolutional(inp, (3, 3, 3, 32), downsample=True, bn=False)  
    x = convolutional(x, (3, 3, 32, 64), downsample=True)   
    x = convolutional(x, (3, 3, 64, 128), downsample=True)  
    x = convolutional(x, (3, 3, 128, 256), downsample=True) 
    x = convolutional(x, (3, 3, 256, 512), downsample=True)  

    for _ in range(2):
        x = residual_block(x, 512, 256, 512)

    x = convolutional(x, (3, 3, 512, 256))  
    x = convolutional(x, (3, 3, 256, 128))  
    x = convolutional(x, (3, 3, 128, 64), bn=False)  

    x = tf.keras.layers.Flatten()(x)  # (bs, 10*10*64)
    x = tf.keras.layers.Dense(units=1024, activation="relu")(x)  
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)   
    logits = tf.keras.layers.Dense(units=1)(x)  
    probs = tf.nn.sigmoid(logits)

    return tf.keras.Model(inputs=inp, outputs=logits)
