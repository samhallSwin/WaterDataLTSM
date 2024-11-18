#Should we generate plots and the interactive map? Takes a while to make 
makePlots = True
makeMap = False

experiment_name = 'only_bendigo_in'

#Target site to predict (label)
targetSiteID = 407254
#target inputs. Can be any number of sites
input_Sites = [407246, 407255, 407229, 407254]

#For each input site, enter any combination of h,f,r for height, flow, rainfall.
input_fields = ['h','h','h']

#How many timesteps into the future we want to predict
time_ahead = 20

timesteps = 70
epochs = 20
units = 70
batch_size = 32
learning_rate = 0.001
