from huggingface_hub import HfApi
api = HfApi()
api.create_repo('imaging-101', repo_type='dataset', private=True)
print('仓库创建成功')


from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='tasks/',
    repo_id='karaoke-monkey/imaging-101',  # 改成这个
    repo_type='dataset',
    allow_patterns='**/fixtures/**',
)
print('上传完成')

from huggingface_hub import HfApi                                                                                                                                                
api = HfApi()                                                                                                                                                                    
api.upload_file(                                                                                                                                                                 
  path_or_fileobj="README_HF.md",                                                                                                                                              
  path_in_repo="README.md",                                                                                                                                                    
  repo_id="HeSunPU/imaging-101-fixtures",                                                                                                                                      
  repo_type="dataset",                                                                                                                                                         
)                                                                                                                                                                                
print('上传完成')
