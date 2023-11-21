# Installation
```
git clone https://github.com/MeshchaninovViacheslav/llama_bot.git
```

# Launch

You should place decoder weights and config in the same folder and call them the same except for the extension. For example, 

- weights
    - decoder.pth
    - decoder.json


```
cd dalleCompressor
python -d path/to/decoder -i path/to/test_image -t text_prompt
```

# Test

There are 10 images of dataset COCO in the folder test_images, you can try them. The PSNR is higher than 40 on average.
