
#%%
from dask.distributed import Client, progress

#%%

if __name__ == "__main__":
    client = Client(threads_per_worker=4, n_workers=1)
    client.cluster.scale(10)
    #client

    #%%
    import time
    import random
    import numpy as np

    def costly_simulation(list_param):
        time.sleep(random.random())
        return sum(list_param), np.mean(list_param)

    # %%
    #%%time 
    costly_simulation([1, 2, 3, 4])

    #%%
    import pandas as pd
    import numpy as np

    input_params = pd.DataFrame(np.random.random(size=(500, 4)),
                                columns=['param_a', 'param_b', 'param_c', 'param_d'])
    input_params.head()


    #%%
    #%%time
    results = []
    for parameters in input_params.values[:10]:
        result = costly_simulation(parameters)
        results.append(result)
        
    results

    # %% 
    #%%time
    ## using dask.delayed for lazy computation occurs where 
    ### 
    import dask

    lazy_results = []



    for parameters in input_params.values[:10]:
        #print(parameters)
        lazy_result = dask.delayed(costly_simulation)(parameters)
        lazy_results.append(lazy_result)
        
    #lazy_results[0]
    #dask.compute(*lazy_results)
    
    # %%
    ## submitting the computation to be done in the background
    futures = dask.persist(*lazy_results)
    results = dask.compute(*futures)
    print(len(results))
    print(results[:5])
    
    ## using futures api
    # dask futures is not lazy and start work immediately
    futures = []
    for parameters in input_params.values:
        future = client.submit(costly_simulation, parameters)
        futures.append(future)
        
    results = client.gather(futures)
    #print(results[:5])
    
    ## map can be used to map function to paraameters
    ##when futures is already called then map will not compute again but 
    ## retrieve the computed results from the future
    #print(input_params.values)
    futures = client.map(costly_simulation, input_params.values)
    results = client.gather(futures)
    #print(results)
    #print(results[0])
    
    ## doing analysis on results
    output = input_params.copy()
    output['result'] = pd.Series(results, index=output.index)
    #print(output.sample(5))
    
    ## for large input parameters of more that 100,000 use Bags
    print("---------- using dask bags ----------------")
    import dask.bag as db
    b = db.from_sequence(list(input_params.values), npartitions=100)
    #print(f"type of b: {type(b)}")
    b = b.map(costly_simulation)
    results_bag = b.compute()
    #print(f"Same results: {np.all(results) == np.all(results_bag)}")
    
    
    #def find_desc(a, b, c):
        
        
    
    #time.sleep(100)
    
"""
Consider scattering large objects ahead of time
with client.scatter to reduce scheduler burden and 
keep data on workers

future = client.submit(func, big_data)    # bad

big_future = client.scatter(big_data)     # good
future = client.submit(func, big_future)  # good
"""