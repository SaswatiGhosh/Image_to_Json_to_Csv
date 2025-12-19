import os
from google import genai
from google.genai import types

# Make sure your GEMINI_API_KEY environment variable is set
client = genai.Client()

def image_to_markdown(image_path, output_filename="output.md"):
    """Converts an image to a markdown file using the Gemini API."""

    print(f"Uploading image: {image_path}")
    # Upload the file to the API
    try:
        uploaded_file = client.files.upload(file=image_path)
        print(f"Uploaded file '{uploaded_file.name}' as '{uploaded_file.mime_type}'")
    except Exception as e:
        print(f"Error during file upload: {e}")
        return

    # Define the prompt to ask for markdown output
    prompt = "Analyze this image and extract all relevant information, text, and structure. Format your entire response as a single, comprehensive Markdown document, preserving layouts and formatting as much as possible."

    print("Generating markdown content with Gemini API...")
    # Generate content with the model
    response = client.models.generate_content(
        model="gemini-2.5-flash", # Use a model that supports multimodal input
        contents=[uploaded_file, prompt],
    )

    # The response text is the markdown content
    markdown_text = response.text

    # Save the text to a .md file
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"Successfully created markdown file: {output_filename}")

    # Clean up the uploaded file from the API after use (optional, files expire in 48h)
    client.files.delete(name=uploaded_file.name)
    print(f"Deleted uploaded file '{uploaded_file.name}' from API.")

def read_md_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Function to convert markdown to CSV using the Gemini API
def convert_md_to_csv_gemini(md_content):
    try:
        # The client automatically picks up the API key from the environment variable
        client = genai.Client()

        # Define the prompt/system instruction to ensure CSV output
        prompt = """
        Convert the following Markdown content into strictly CSV format. 
        Ensure proper quoting and separation of fields. Do not include any introductory or concluding text, only the CSV data.
        """
        
        # Call the API with the content and the instruction
        response = client.models.generate_content(
            model="gemini-2.5-flash", # A suitable model for text processing
            contents=[prompt, md_content],
        )

        return response.text

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Replace 'path/to/your/sample.jpg' with the actual path to your image
    image_file_path = "C:/Users/saswati/Desktop/MLOps_DevOps/Image_to_csv/image/image.png" 
    md_file_path = "C:/Users/saswati/Desktop/MLOps_DevOps/Image_to_csv/output/image_content.md"  # Replace with the path to your .md file
    csv_file_path = "C:/Users/saswati/Desktop/MLOps_DevOps/Image_to_csv/output/output.csv"  # Desired output .csv file path
    image_to_markdown(image_file_path, md_file_path)
    if os.path.exists(md_file_path):
        md_content = read_md_file(md_file_path)
        if md_content:
            csv_data = convert_md_to_csv_gemini(md_content)
            if csv_data:
                # Save the result as a CSV file
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
                    f.write(csv_data)
                print(f"Successfully converted '{md_file_path}' to '{csv_file_path}'.")
    else:
        print(f"Error: The file '{md_file_path}' was not found.")
