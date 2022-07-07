# Machine Learning Assignment 2 on Image quantization 





##### Submitted by Lalit (MCS - 22) and Vidhi Khare (MCS - 54)

##### Submitted To : Prof. Vasudha Bhatnagar





## Introduction

As per the given problem we have to take two picture one of outdoor location and second of an object on the wall of our home and create a codebook and quantize the image.


We have to create 4 bit, 8 bit and 12 bit codebooks.

Generally , in a colored image , each pixel is of size 3 bytes (RGB) where color can have intensity values from 0 to 255.
As we know that 4 bit has 2<sup>4</sup> = 16 colors and similarly for 8 bit and 12 which has 256 and 4096 colors.
	
Quantization is a lossy compression technique achieved by compressing a range of values to a single quantum value. When the number of discrete symbols in a given stream is reduced, the stream becomes more compressible. For example, reducing the number of colors required to represent a digital image makes it possible to reduce its file size.

## Approach

Color quantization can be done using clustering  where each of the color pixels will be grouped into clusters and each cluster will then be represented as a unique color in the new image. And here we are using __KMeans__.



## Tools and packages used

- python jupyter
- numpy
- MiniBatchKMeans, from sklearn
- matplotlib
- Pillow (for image loading saving etc.)



## MiniBatchKMeans

Mini Batch K-means algorithm‘s is a form a KMeans where the main idea is to use small random batches of data of a fixed size, so they can be stored in memory. Each iteration a new random sample from the dataset is obtained and used to update the clusters and this is repeated until convergence.



## Reason of using MiniBatchKMeans

Reason for using MiniBatchKMeans over KMeans is simply that for 8 Bits and 12 Bits Quantization, KMean is very expensive in terms of temporal and spatial cost.



## Procedure

- Going to get the images and then load those in our project and then will convert them into an np array (in 8 bit color image)
- Get the original shape of the image ie width, height and depth
- Reshaping the image array
- Use MiniBatchKMeans forming clusters and no of clusters is no of colors 
- Recreating the image back and write the obervations.



## Code 

```python

#Defining the number of bits to be used and number of colors to be used
#n_bits = 4, 8, 12
n_bits = int(input("Enter the number of bits to be used (4, 8, 12): "))
n_colors = 2**n_bits

# Loading the image
image_name = input("Enter the name of the image to be Quantized: ")
image = Image.open(image_name)

# Convert to floats instead of the default 8 bits integer coding. 
# Dividing by 255 is important so that plt.imshow behaves works well on float data (need to be in the range [0-1])
image = np.array(image, dtype=np.float64)/255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(image.shape)
image_array = np.reshape(image, (w * h, 3))

print("---------- Fitting model ----------")
kmeans = MiniBatchKMeans(n_clusters=n_colors, random_state=0).fit(image_array)

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
labels = kmeans.predict(image_array)

# Recreate the (compressed) image from the code book & labels
def recreate_image(codebook, labels, w, h):
    return codebook[labels].reshape(w, h, -1)

# Displaying and Saving the Quantized image
plt.axis("off")
print(f"Quantized image ({n_colors} colors, K-Means)")
plt.imsave("qti1.jpg", recreate_image(kmeans.cluster_centers_, labels, w, h))
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

```



__Running the above the code for both the images and for all three options__



### Image 1 (Outdoor image of a Railway Station)


#### Original Image

![outdoor_img](Images/Outdoor.jpg)



This is the original image and the details of elements are quite visible in this image.

Dimensions of the image are 1040 x 780 px , taking size 2.50 MB



#### 4 Bits 



Now lets look at 4 bits quantized image 

![outdoor_4](Images/qto1.jpg)



__Observations__ : The difference is clearly visible here, Look at the sky, there is a clear distinction between the different shades of the colors of the sky. And why so, the reason is simple in original image we had somewhat around 96,615 colors but here we have quantized the image into 4 bit, that is 16 colors.

The same difference can be seen on floor in the image. There are different shades of grey and black in the original picture and there the dust kind of particles can be seen but in quantized image these are not visible clearly and here only two shades of grey are visible.

__Codebook__ 

```python
array([[0.93726658, 0.9719419 , 0.97634523],
       [0.32478721, 0.24776073, 0.17619611],
       [0.59194273, 0.57783967, 0.55052484],
       [0.14667704, 0.11229013, 0.10231122],
       [0.72666404, 0.73279874, 0.71828243],
       [0.87482564, 0.91293266, 0.92648045],
       [0.46281561, 0.43701454, 0.40398189],
       [0.27173352, 0.27174446, 0.27142382],
       [0.22234853, 0.20654557, 0.19263601],
       [0.65003691, 0.66176019, 0.64855794],
       [0.39052181, 0.35158228, 0.31558428],
       [0.85440819, 0.67939137, 0.54883467],
       [0.31372736, 0.1425629 , 0.13139668],
       [0.79085646, 0.85893306, 0.91240734],
       [0.17942231, 0.14958037, 0.13338987],
       [0.50097124, 0.49017918, 0.47317422]])
```



#### 8 Bits

![outdoor_8](Images/qto2.png)



__Observation__ : The difference is almost unnoticable. Because now we have 256 colors. We can see some difference in sharpness but it is almost unnoticable but, we can find some differences when we zoom and analyse different area's of the image. 

For example see the LED display on pole in the original it is more visible than that in the quantized image.

__Codebook__ 

```python
array([[0.82875165, 0.89547267, 0.95820758],
       [0.30100649, 0.22996485, 0.16356342],
       [0.56185826, 0.55431763, 0.53745829],
       ...,
       [0.4523981 , 0.45476341, 0.46058318],
       [0.8442741 , 0.89956207, 0.94962243],
       [0.7105037 , 0.71831447, 0.71072647]])
```



#### 12 Bits

![outdoor_12](Images/qto3.jpg)



__Observation__ : Again difference is almost unnoticable but if we would look in deep, then we may find some differences. To differentiate, we can observe the size of both the images which are clearly not equivalent, this is due to compression that happened while quantizing the image. (Quantized Image = 440 KB, Original Image = 1.88 MB)

__Codebook__ 

```python
array([[0.90784314, 0.9620915 , 0.99281046],
       [0.79934641, 0.85424837, 0.89738562],
       [0.18823529, 0.16078431, 0.1372549 ],
       ...,
       [0.18980392, 0.14431373, 0.1372549 ],
       [0.32941176, 0.2627451 , 0.19215686],
       [0.75222816, 0.82388592, 0.90196078]])
```



### Image 2 (Indoor image of a Calendar on a wall)


#### Original Image

![indoor_img](Images/Indoor.jpg)



This is the original image and the details related to numbers, diffrent characters and their respective colors along with different shades and patterns are quite visible in this image. 

Dimensions of the image are 1788 x 1788 px , taking size 1.88 MB



#### 4 Bits 



Now lets look at 4 bits quantized image 

![indoor_4](Images/qti1.jpg)



__Observations__ : The difference is clearly visible here, Look at the wall the different shade of color white are not clearly visble here, and also the shadow of the calendar with black-grey color is now having different patches of color ditinctively visible.

In other parts of the calendar too the colors are now dull as compared to the original image and the shades of pink and light yellow are not visible at all. And why so, the reason is same as above, the original image that we had was composed of several colors but here we have quantized the image into 4 bit, that is only 16 colors. 

But, as compared to the outdoor image quantization for 4 Bits, here, we do not see siginificant quality drop and compression as compared to the original indoor image, this is due to the reason that outdoor image had more color shades whereas here, the image never had that many colors to begin with.

__Codebook__ 

```python
array([[0.78076694, 0.76355564, 0.7414921 ],
       [0.37632788, 0.33785516, 0.30356201],
       [0.6558868 , 0.61992363, 0.59130568],
       [0.16446724, 0.13415107, 0.10772411],
       [0.05189985, 0.35420003, 0.73018466],
       [0.73262767, 0.70443735, 0.67245428],
       [0.79372106, 0.68820899, 0.24116885],
       [0.45426346, 0.15840114, 0.15903256],
       [0.47496465, 0.42510148, 0.39868031],
       [0.44887361, 0.55388696, 0.67928409],
       [0.54673653, 0.23738874, 0.22914348],
       [0.04840409, 0.27317529, 0.57215391],
       [0.81878979, 0.81514184, 0.81948275],
       [0.62230163, 0.69375663, 0.77574612],
       [0.55807078, 0.51595299, 0.49261033],
       [0.27732266, 0.24068791, 0.20107717]])
```



#### 8 Bits

![indoor_8](Images/qti2.jpg)



__Observation__ : The differences here are also almost unnoticable as was the case with outdoor image quantized with 256 colors (8 bits). We can see some difference in areas that were exposed to more light as compared to there surrounding but it is almost unnoticable but, we can find difference when we zoom and analyse different area of the image. 

For example see the area of the two red strips and a green strip on the left, there are few places which are in darker shade of there respective colors as compared to the original. This change in color is only visible after very close oservation of both the images.

__Codebook__ 

```python
array([[0.68549135, 0.66162301, 0.62009528],
       [0.28637873, 0.23712233, 0.19305301],
       [0.79483325, 0.7560939 , 0.77680287],
       ...,
       [0.801444  , 0.61112649, 0.59092184],
       [0.36070604, 0.44580446, 0.5362373 ],
       [0.76376365, 0.6936304 , 0.67627486]])
```



#### 12 Bits

![indoor_12](Images/qti3.jpg)



__Observation__ : Difference is unnoticeable here even after very close observations. To differentiate, we can observe the size of both the images which are clearly not equivalent, this is due to compression that happened while quantizing the image. (Quantized Image = 440 KB, Original Image = 1.88 MB)

__Codebook__ 

```python
array([[0.80392157, 0.81764706, 0.83627451],
       [0.43333333, 0.38235294, 0.32941176],
       [0.65294118, 0.64183007, 0.62810458],
       ...,
       [0.77254902, 0.79215686, 0.77647059],
       [0.78823529, 0.78431373, 0.77647059],
       [0.59869281, 0.56862745, 0.52679739]])
```
