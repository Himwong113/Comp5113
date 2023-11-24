import requests
import pathlib

# URL of the image you want to download



# Send an HTTP GET request to the image URL

cnt=1100

while True:

    # Define the number of strings you want to generate
    # Adjust this to the desired number of strings

    id = str(cnt).zfill(12)

    image_url = "http://images.cocodataset.org/val2017/" + id+".jpg"

    response = requests.get(image_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the content of the response, which is the image binary data
        image_data = response.content

        # Specify the file path where you want to save the image
        root = pathlib.Path().absolute()

        print(root)
        file_path = str(root)+"\\traindata\\"+id+".jpg"
        print(file_path)

        # Open the file in binary write mode and write the image data to it
        with open(file_path, "wb") as file:
            file.write(image_data)

        print(f"Image downloaded and saved to {file_path}")
    else:
        print(f"{response.status_code}")
        #print(f"Failed to download the image. Status code: {response.status_code}")

    if cnt % 100 == 0:
        print(cnt)
    if cnt>10000:
        break

    cnt+=1