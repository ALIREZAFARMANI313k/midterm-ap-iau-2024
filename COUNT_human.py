#authors:alireza farmani, alimohammad faghedi heris , yousef haji ali labbaf


import cv2

# Load the pre-trained human detection model
human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

def count_humans(image_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect humans in the image
    humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Print the number of humans detected

    print("Number of humans detected:", len(humans))

# Call the function with the image path

count_humans('photo.jpg')




#توضیحات :این کد با همکاری علیرضا فرمانی وعلیمحمد فاقدی هریس و یوسف حاجی علی لباف ساخته شده است 

#این کد تعداد انسان های موجود در یک تصویر را میشمارد وان را در خروجی چاپ میکند 

#  قابل ذکر است برای شمارش درست کافی است که  فایل ایکس ام ال موجود در گیت هاب را دانلود کرده و سپس هم کد و هم این فایل و فایل عکس را در یک پوشه قرر داده و سپس برنامه را اجرا کنیم


