# **Image Captioning**
- Image Captioning Model 
- Based on standard CNN LSTM architecture ( will add attention later too)
- Utilized Pretrained models to extract image features 
- Used Flickr8k dataset
- It works quite simply We Input the image and it outputs descriptive captions
## **Image Captions** <br>
Some examples of Images-Caption pairs in the flickr dataset <br>
<br>
<img src="Images/1012212859_01547e3f17.jpg" alt="Alt text" width="300"> <br>
<em>**Caption**</em> - White dog playing with a red ball on the shore near the water . <br> 

<img src="Images/1000268201_693b08cb0e.jpg" alt="Alt text" width="300"> <br>
<em>**Caption**</em> - A child in a pink dress is climbing up a set of stairs in an entry way <br>

## **Image Features Extraction** 

- Extracted image features using the **VGG** 16 model
- loaded the VGG16 model with the necessary imports
-  restructured the model by Using `model.layers[-2].output` and extracted features from the penultimate layer, omitting the final activation layer, to ensure raw feature representations are obtained as the model was initially designed for classification


```python
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
for img_name in tqdm(os.listdir(directory)):
    img_path = directory + '/' + img_name
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    image_id = img_name.split('.')[0]
    features[image_id] = feature
```

## **Create Mapping of Captions to Images**

- Created mapping of captions to images for better preprocessing of text captions data

```python
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)
```

## **Preprocessing text data**
- preprocessed text captions ( cleaning textual data)
```python
def Preprocess_Captions(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'start ' + " ".join([word for word in caption.split() if len(word)>1]) + ' end'
            captions[i] = caption
```

## **Tokenizing text captions**

- Used the keras tokenizer to tokenize text captions <br>
```python 
from tensorflow.keras.preprocessing.text import Tokenizer
Tokenizer = Tokenizer()
Tokenizer.fit_on_texts(all_captions)
vocab_size = len(Tokenizer.word_index) + 1
```
