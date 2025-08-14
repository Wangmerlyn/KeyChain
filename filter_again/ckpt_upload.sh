sas_token="sp=racwdl&st=2025-07-30T19:00:39Z&se=2025-08-06T03:15:39Z&skoid=7b3a9ac3-4eaa-434a-8801-b2b90159bf0b&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2025-07-30T19:00:39Z&ske=2025-08-06T03:15:39Z&sks=b&skv=2024-11-04&spr=https&sv=2024-11-04&sr=c&sig=Xw7mffXV4W5zmVPpngZMxs1ByZ8RZw0m9cPzOKYpeog%3D"
ckpt_path=/home/aiscuser/LongContextDataSynth/filter_again/eval_results_420
echo "🔄 Uploading checkpoints to Azure Blob Storage..."
echo "target path: https://sanbpx4p3idss6q.blob.core.windows.net/longcontext/models/siyuan/test_code/longcontext_syth/filter_again/?${sas_token}"
azcopy copy --recursive --overwrite="false" $ckpt_path "https://sanbpx4p3idss6q.blob.core.windows.net/longcontext/models/siyuan/test_code/longcontext_syth/filter_again/?${sas_token}"