import zstandard as zstd

input_file = r"D:\下载\pusht_expert_train.h5.zst"
output_file = r"C:\Users\RaymondZh\Desktop\github\le-wm\datasets\pusht.h5"

with open(input_file, 'rb') as compressed:
    dctx = zstd.ZstdDecompressor()
    with open(output_file, 'wb') as destination:
        dctx.copy_stream(compressed, destination)
print("解压完成！")