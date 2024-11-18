#Should we generate plots and the interactive map? Takes a while to make 
makePlots = True
makeMap = False
predictions2csv = True

experiment_name = 'test_others'

#Target site to predict (label)
targetSiteID = 407254
target_fields = ["height", "flow"] 

#target inputs. Can be any number of sites
input_Sites = [
    {"site_id": "407246", "fields": ["height", "flow"]},
    {"site_id": "407255", "fields": ["height", "flow"]},
    {"site_id": "407229", "fields": ["height", "flow"]},
    {"site_id": "407254", "fields": ["height", "flow"]},
]


#How many timesteps into the future we want to predict
time_ahead = 20

timesteps = 50
epochs = 10
units = 50
batch_size = 32
learning_rate = 0.001
