from inhs_outlining import *

import multiprocessing as mp
import pathlib
import random
import shutil
import os

def get_fifteen_species():
    target_species = [species
                      for species, count in Fish.count_fish_per_species().items()
                      if count >= 100]
    target_fish = []
    for species in target_species:
        all_of_species = Fish.all_of_species(*species.split(' '))
        random.shuffle(all_of_species)
        target_fish += all_of_species[:100]
    return target_fish


def get_seven_genera(max_fish_per_genus):
    target_genera = [genus
                     for genus, count in Fish.count_fish_per_genus().items()
                     if count >= 153]
    target_fish = []
    for genus in target_genera:
        all_of_genus = Fish.all_of_genus(genus)
        random.shuffle(all_of_genus)
        target_fish += all_of_genus[:max_fish_per_genus]
    return target_fish


def generate_frags(frags_dir, fishes, labeler, ids = None):
    nproc = mp.cpu_count()
    groups = [list(fish_group) for fish_group in np.array_split(fishes, nproc)]
    processes = []

    def generate_frag(fishes, procindex):
        pathlib.Path(frags_dir).mkdir(exist_ok=True)
        with open(f"{frags_dir}/{procindex}", 'w') as frag:
            for fish in fishes:
                efds, locus = fish.encoding
                row = np.append(locus, efds.ravel()).astype(str)
                if ids is None:
                    frag.write(','.join(row) + (f",{labeler(fish)}\n"))
                else:
                    frag.write(','.join(row) + (f",{labeler(fish)},{ids(fish)}\n"))
                del fish

    for n in range(nproc):
        p = mp.Process(target=generate_frag, args=(groups[n], n))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()


def build_mat_from_frags(name, frags_dir):
    mat = []
    labels = []
    for fragpath in pathlib.Path(frags_dir).iterdir():
        with open(fragpath, 'r') as frag:
            for row in frag.readlines():
                cols = row.split(',')
                #If try is valid, it means it means the last column is an ID (int), otherwise it's a label (string)
                #As a result, we must split the content and its labels accordingly
                try:
                    int(cols[-1])
                    mat.append(cols[:-2])
                    labels.append(",".join(cols[-2:]).strip())    
                except ValueError:
                    mat.append(cols[:-1])
                    labels.append(cols[-1].strip())
    
    mat = pad_ragged(mat)
    mat = np.concatenate((mat, np.array(labels).reshape(-1, 1)), axis=1)
    np.savetxt(f"{name}.csv", mat, fmt='%s', delimiter=',')
    shutil.rmtree(frags_dir)

#When generating ids, add "lambda fish: fish.id" to parameters in generate_dataset call
def generate_dataset(name, fishes, labeler, ids = None):
    frags_dir = f"{name}_frags/"
    generate_frags(frags_dir, fishes, labeler, ids = ids)
    build_mat_from_frags(name, frags_dir)


#ajani's additions:
# Theory: try getting the mat just in its regular form, with or without the 0s - then, when given a padded diva, it builds a mat w/o the 0s? i didn't think this through
# do i just test it out? like input random nums? or add another function that just generates a bunch of files...idk...
# maybe a function that takes in a pad amount somehow, let's reassess our goal
# to generate a bunch of datasets and test run_cv_trials on them..
# how about we DO NOT return them as an np.array and just try a whole bunch of nums for it- only thing we would have to modify is generate dataset and buildmatfromfrags 
# So do we wanna do that? just add an additional param for lengths to test? i guess we can - make a new func for build mat whihc just takes length from generate dataset call..
# In build mats: copy allthat was b4 but for each item in length -> run value_ragged for the length, concatenate w/ labels and save txt in that tree
# Path: testingdatasets/min(lenght)_max(length)/1mm_seven_genera_;ength.csv - majority of them not gonna be able to reshape when we do classification anyway...try/except for that
# padded = lambda row, amt: 

# Same as "build_mat_from_frags" except for addition of lengths parameter
# lengths is iterable - for each length x (int), it will generate a file  
# containing descriptors that all have the length of x
# * Files saved in separate dir in testing_datasets:
#   Notation: 'smallest number in lengths'_'largest number in lengths'  
# TODO: someway to check for mean on outside....outsideoutside....
def build_mats(name, frags_dir, lengths):
    mat = []
    labels = []
    for fragpath in pathlib.Path(frags_dir).iterdir():
        with open(fragpath, 'r') as frag:
            for row in frag.readlines():
                cols = row.split(',')
                #If try is valid, it means it means the last column is an ID (int), otherwise it's a label (string)
                #As a result, we must split the content and its labels accordingly
                try:
                    int(cols[-1])
                    mat.append(cols[:-2])
                    labels.append(",".join(cols[-2:]).strip())    
                except ValueError:
                    mat.append(cols[:-1])
                    labels.append(cols[-1].strip())
    path = f"testing_datasets/{min(lengths)}_{max(lengths)}/"
    os.makedirs(path, exist_ok=True)
    for length in lengths:
        temp_mat = mat.copy()
        temp_mat = value_ragged(temp_mat, length)
        temp_mat = np.concatenate((temp_mat, np.array(labels).reshape(-1, 1)), axis=1)
        np.savetxt(f"{path}/{name}_{length}.csv", temp_mat, fmt='%s', delimiter=',')
    shutil.rmtree(frags_dir)

# Same as "generate_datasets" but it takes in length values for build_mats
def generate_datasets(name, fishes, labeler, lengths, ids = None):
    frags_dir = f"{name}_frags/"
    generate_frags(frags_dir, fishes, labeler, ids = ids)
    build_mats(name, frags_dir, lengths)

if __name__ == "__main__":
    #generate_dataset("1mm_fifteen_species_median_no_id", get_fifteen_species(), lambda fish: f"{fish.genus} {fish.species}")
    #generate_dataset("1mm_seven_genera", get_seven_genera(153), lambda fish: fish.genus)
    #generate_dataset("testing_datasets/1mm_aug_seven_genera_mean_no_id", get_seven_genera(250), lambda fish: fish.genus)
    
    generate_datasets("1mm_seven_genera", get_seven_genera(153), lambda fish: fish.genus, (56, 82, 73))