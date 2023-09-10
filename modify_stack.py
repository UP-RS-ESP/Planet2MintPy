import h5py

filename = "/raid-manaslu/amueting/PhD/Project3/PlanetScope_Data/aoi7/new_weights/mintpy/inputs/geo_offsetStack_aoi7.h5"
# filename = "/home/ariane/Documents/Project3/PlanetScope_Data/aoi7/test/geo_offsetStack_aoi7.h5"
    
f = h5py.File(filename, 'r+')

data1 = f['azimuthOffsetStd']     
data2 = f['azimuthOffsetStd']   


for i, d in enumerate(data1):
    data1[i] = ((d / 100)**4)*100
    
for i, d in enumerate(data2):
    data2[i] = ((d / 100)**4)*100


f.close()                       
