
#%%
import PIL
from feat import FeatureExtractor, ImgPropertySetReturnType, load_model_and_preprocess
import multiprocessing
from glob import glob
import dask
from dask.distributed import Client, progress
import time
import asyncio



def get_imgs_and_extract_features_multiprocess(img_path, img_resize_width,
                                               img_resize_height,
                                               model_family, model_name,
                                               img_normalization_weight,
                                               seed, #model, preprocess, #images_list, features_list, 
                                               #model_artefacts_dict, #lock
                                               ):
    #lock.acquire()
    feat_extract = FeatureExtractor(seed=seed, img_resize_width=img_resize_width,
                                    img_resize_height=img_resize_height, 
                                    model_family=model_family,
                                    model_name=model_name,
                                    img_normalization_weight=img_normalization_weight
                                    )
    #images_list = []
    #features_list = []
    feat_extract.set_seed_consistently()
    #model = model_artefacts_dict['model']
    #preprocess = model_artefacts_dict['preprocess']
    model, preprocess = feat_extract.load_model_and_preprocess_func()
    feature_extractor = feat_extract.get_feature_extractor(model)
    img = feat_extract.load_and_resize_image(img_path, img_resize_width, img_resize_height)
    
    img_for_infer = feat_extract.load_image_for_inference(img_path, feat_extract.image_shape)
    feature = feat_extract.extract_features(img_for_infer, feature_extractor, preprocess)
    #images_list.append(img)
    #features_list.append(img_for_infer)
    #print(f"total imgs processed {len(images_list)}")
    #print(f"total features processed {len(features_list)}")
    #lock.release()
    #return list(images_list), list(features_list)
    #print(f"processed: {img_path} ")
    return img, feature



#%%



def img_feature_extraction_implementor(img_property_set,
                                       feature_extractor_class = None,
                                       seed=2024, img_resize_width=224,
                                       img_resize_height=224,
                                       model_family="efficientnet",
                                       model_name="EfficientNetB0",
                                       img_normalization_weight="imagenet",
                                       use_cropped_imgs=True,
                                       ):
    client = Client(threads_per_worker=4, n_workers=1)
    client.cluster.scale(10)
    
    img_paths = sorted(img_property_set.img_paths)
    
    #manager = multiprocessing.Manager()
    #images_list = manager.list()
    #features_list = manager.list()
    #model_artefacts_dict = manager.dict()
    
    image_shape=(img_resize_height, img_resize_width, 3)
    
    """model, preprocess = load_model_and_preprocess(input_shape=image_shape, model_family=model_family, 
                                                    model_name=model_name, 
                                                    weight=img_normalization_weight,
                                                    )"""
    #await model, preprocess
    
    #model_artefacts_dict['model'] = model
    #model_artefacts_dict['preprocess'] = preprocess
    #print(type(model_artefacts_dict))
    args_for_multiprocess = [(img_path, img_resize_width, img_resize_height,
                              model_family, model_name, 
                              img_normalization_weight, seed, #model, preprocess, #images_list,
                              #features_list, #model_artefacts_dict, #lock
                              )
                             for img_path in img_paths
                             ]
    """num_processes = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(num_processes) as pool:
    
        print("waiting for multiprocess to finish")
        results = pool.starmap(get_imgs_and_extract_features_multiprocess, args_for_multiprocess)
    print(type(results)) 
    print(results)
    """
    """
    Consider scattering large objects ahead of time
    with client.scatter to reduce scheduler burden and 
    keep data on workers

    future = client.submit(func, big_data)    # bad

    big_future = client.scatter(big_data)     # good
    future = client.submit(func, big_future)  # good
    """    
    dask_result = []
    
    for param in args_for_multiprocess:
        #res_dayed = dask.delayed(get_imgs_and_extract_features_multiprocess)(*param)
        #dask_result.append(res_dayed)
        #big_future = client.scatter(param)
        dask_res = client.submit(get_imgs_and_extract_features_multiprocess, *param)
        dask_result.append(dask_res)
    
    res = client.gather(dask_result)
    #bg_compute = dask.persist(*dask_result)    
    #res = dask.compute(*bg_compute)
    imgs = []
    features = []
    for items in res:
        img, feature = items
        imgs.append(img)
        features.append(feature)
        
    img_property_set.imgs = imgs
    img_property_set.features = features
    
    print(f"full res type is : {type(res)}")
    print(f"full res length: {len(res)}")
    #imgs, features = res
    print("------------- res ------------------/n")
    print(f"res index 0 type: {(res[0])}")
    print(f"length of img_paths: {len(img_paths)}")
    
    #img_property_set.imgs = imgs
    #img_property_set.features = features
    #images_list_re, features_list_re = results
    
    #img_property_set.imgs = list(images_list_re)
    #img_property_set.features = list(features_list_re)
    
    print(f"num of images: {len(img_property_set.imgs)}")
    print(f"num of features: {len(img_property_set.features)}")
    
    return img_property_set
    
    

#%%
img_dir = "/home/lin/images"
img_paths_list = sorted(glob(f"{img_dir}/*"))   

img_property_set = ImgPropertySetReturnType(img_paths=img_paths_list, img_names="xxx", total_num_imgs=100, max_num_clusters=4)

if __name__ == '__main__':
    #client = Client(threads_per_worker=4, n_workers=1)
    #client.cluster.scale(10)
    img_feature_extraction_implementor(img_property_set=img_property_set,
                                   use_cropped_imgs=False
                                   )
    #time.sleep(100)
# %%

