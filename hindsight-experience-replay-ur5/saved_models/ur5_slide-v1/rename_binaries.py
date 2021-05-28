import os 


FolderName = "/home/stefan/Documents/Masterarbeit/DRL/hindsight-experience-replay-ur5/saved_models/ur5_slide-v1"
#os.rename("rename.txt","rename_123.txt")
lstBinaries = sorted(os.listdir(FolderName))

lstBinaries.remove("model.pt")
lstBinaries.remove("rename_binaries.py")
for i,binary in enumerate(lstBinaries):
    new_name = binary.split("_")[0] + "_" + str(i+1) + ".pt"
    os.rename(binary, new_name)
    #print(binary)