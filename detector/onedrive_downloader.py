import argparse

parser = argparse.ArgumentParser(description="Download a file from OneDrive")
parser.add_argument("url", help="URL of the file to download")
parser.add_argument("filename", help="Name of the file to save as")
args = parser.parse_args()


def download(url, filename):
    import requests

    url += "&download=1"
    print("Downloading from", url, "to", filename)
    response = requests.get(url)
    if response.status_code == 200:
        print("Downloaded", len(response.content), "bytes")
        content = response.content

        # save the content to a file
        with open(filename, "wb") as file:
            file.write(content)
        print("Saved to", filename)
        return True
    else:
        print("Error:", response.status_code)
        return False


download(args.url, args.filename)
