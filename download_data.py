from huggingface_hub import snapshot_download

print("Starting download... This may take a few minutes for 2.6 GB.")
snapshot_download(
    repo_id="Ha-Te-21/tigrinya-corpus", 
    repo_type="dataset",
    local_dir="./tigrinya-corpus"
)
print("Download complete! Look inside the './tigrinya-corpus' folder.")
