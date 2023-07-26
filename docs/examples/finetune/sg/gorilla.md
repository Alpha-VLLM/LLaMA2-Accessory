## Demos of fine-tuned LLaMA2 using the gorilla tensorflow dataset.


### **Instruction 1:**
I am working on a project where I need to cluster similar images of street art. How can I get the relevant feature vectors from these images for clustering?

**Answer:**

<<\<domain\>>>: Image feature vector, <<\<api_call\>>>: hub.KerasLayer('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4'), <<\<api_provider\>>>: TensorFlow Hub, <<\<explanation\>>>: This API call loads a pre-trained MobileNetV2 model from TensorFlow Hub, which has been trained on the ImageNet dataset. This model is used for extracting feature vectors from images. You can feed your street art images into this model to obtain feature vectors, which can then be used for clustering. 

<<\<code\>>>: 
```
import tensorflow as tf
import tensorflow_hub as hub
model = tf.keras.Sequential([
    hub.KerasLayer('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4',
                   input_shape=(224, 224, 3),
                   trainable=False)
])

image = tf.keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(224, 224))
image_array = tf.keras.preprocessing.image.img_to_array(image)
image_array = tf.expand_dims(image_array, 0)

features = model.predict(image_array)
```

### **Instruction 2:**
My dog is always getting into things that can be dangerous. Can you help me identify the object my dog is about to eat?

**Answer:**

<<\<domain\>>>: Image object detection, <<\<api_call\>>>: detector = hub.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'), <<\<api_provider\>>>: TensorFlow Hub, <<\<explanation\>>>: This API call loads a pre-trained TensorFlow Hub model for object detection in images using the SSD MobileNet V2 architecture. Once the model is loaded, it can be used to detect objects in images, allowing you to identify the object your dog is about to eat. The model is trained on the COCO dataset, which includes a variety of common objects. 

<<\<code\>>>: 
```
import tensorflow_hub as hub
detector = hub.load('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')
```

### **Instruction 3:**
For arranging my documents, I want to compute the semantic similarity between a set of sentences.

**Answer:**

<<\<domain\>>>: Text embedding, <<\<api_call\>>>: hub.load('https://tfhub.dev/google/universal-sentence-encoder/4'), <<\<api_provider\>>>: TensorFlow Hub, <<\<explanation\>>>: The API call loads the Universal Sentence Encoder model from TensorFlow Hub. This model is designed to encode text into high-dimensional vectors, which can be used for various natural language tasks, including text classification, semantic similarity, clustering, etc. By using this model, you can compute the semantic similarity between a set of sentences, which can help you arrange your documents based on their content. 

<<\<code\>>>: 
```
import tensorflow_hub as hub
embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
```