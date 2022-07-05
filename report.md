# Machine Learning Assignment 2 on Image quantization 





##### Submitted by Lalit (MCS - 22) and Vidhi Khare (MCS - 54)

##### Submitted To : Prof. Vasudha Bhatnagar





## Introduction

As per the given problem we have to take two picture one of outdoor location and second of indoor object and create a codebook and quantize the image.


We have to create 4 bit , 8 bit and 12 bit codebooks.

Generally , in a colored image , each pixel is of size 3 bytes (RGB) where color can have intensity values from 0 to 255.
As we know that 4 bit has 2<sup>4</sup> = 16 colors and similarly for 8 bit and 12 which has 256 and 4096 colors.
	
Quantization is a lossy compression technique achieved by compressing a range of values to a single quantum value. When the number of discrete symbols in a given stream is reduced, the stream becomes more compressible. For example, reducing the number of colors required to represent a digital image makes it possible to reduce its file size.

## Approach

Color quantization can be done using clustering  where each of the color pixels will be grouped into clusters and each cluster will then be represented as a unique color in the new image. And here we are using __KMeans__.



## Tools and packages used

- python jupyter
- numpy
- KMeans , from sklearn
- matplotlib
- Pillow (for image loading saving etc.)

## Procedure

- Going to get the images and then load those in our project and then will convert them into an np array (in 8 bit color image)
- Get the original shape of the image ie width, height and depth
- Reshaping the image array
- Use KMeans forming clusters and no of clusters is no of colors 
- Recreating the image back and write the obervations.



## Code 

```python
n_bit = 4, 8 , 12
n_colors = 2**n_bit

# Load the Summer Palace photo
image = Image.open("outdoor.jpg")

# Convert to floats instead of the default 8 bits integer coding. Dividing by
# 255 is important so that plt.imshow behaves works well on float data (need to
# be in the range [0-1])
image = np.array(image, dtype=np.float64)/255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(image.shape)
image_array = np.reshape(china, (w * h, 3))

print("Fitting model ")
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array)

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
labels = kmeans.predict(image_array)



def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    return codebook[labels].reshape(w, h, -1)


# # Display all results, alongside original image

plt.figure(2)
plt.clf()
plt.axis("off")
plt.title(f"Quantized image ({n_colors} colors, K-Means)")
plt.imsave("qti.jpg",recreate_image(kmeans.cluster_centers_, labels, w, h))

```



__Running the above the code for both the images and for all three options__



### Image 1 (Outdoor image of a Railway Station)



#### Original Image

![img](http://192.168.43.1:4444/q/cyy3)



This original image and details are quite visible in this image. Size of the image is 1040x780 px taking size 198.6 kb



#### 4 Bits 



Now lets look 4 bit quantize image 

![outdoor_4](C:\Users\Lenovo\Documents\outdoor_4.jpg)



__Observations__ : The difference is clearly visible here, Look at the sky , there is clear distinction between the different shades of the colors of the sky. And why so, the reason is simple in above image we have somewhat around 96,615 colors but here we have quantized the image into 4 bit , that is 16 colors.

The same difference can be seen on floor in the image. There is different shades of grey and black in the original picture and there the dust kind of particles can be seen but in quantized image these are not visible clearly and there only two shades of grey is visible here.

__Codebook__ 

```python
array([[0.3223426 , 0.24592218, 0.17529732],
       [0.78712174, 0.85432532, 0.90667613],
       [0.47764182, 0.4596491 , 0.43585597],
       [0.15251028, 0.11741676, 0.10603203],
       [0.63644951, 0.64417844, 0.62927175],
       [0.94124689, 0.9739545 , 0.97822212],
       [0.39496972, 0.35731082, 0.32094564],
       [0.90907283, 0.81005503, 0.68769283],
       [0.26216115, 0.2610399 , 0.26019431],
       [0.85284311, 0.91276604, 0.95627993],
       [0.31143346, 0.14136356, 0.13131114],
       [0.71412018, 0.72267153, 0.71254054],
       [0.56905868, 0.55607564, 0.5326591 ],
       [0.91032138, 0.91559276, 0.88173225],
       [0.82479528, 0.64079125, 0.51565052],
       [0.19362266, 0.16895232, 0.15183069]])
```



#### 8 Bits

![outdoor_4](C:\Users\Lenovo\Documents\8bit.png)



__Observation__ : The difference is almost unnoticable. Because now we have 256 colors. We can see some difference in sharpness

but it is almost unnoticable but we can find difference when we zoom and analyse different area on image. For example see the LED display on pole in original it is more visible.



#### 12 Bits

![IMG20220409170254](C:\Users\Lenovo\Documents\IMG20220409170254.jpg)



__Observation__ : Again difference is almost unnoticable but if we would look in deep then we may find notice .



## Image 1 (Indore image of object on wall )







