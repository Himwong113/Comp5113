import image

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

# Define the number of strings you want to generate
num_strings = 10  # Adjust this to the desired number of strings

# Generate and print the padded strings
for i in range(1, num_strings + 1):
    padded_string = str(i).zfill(12)
    print(padded_string)
