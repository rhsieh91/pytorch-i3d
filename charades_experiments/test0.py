import os

if __name__ == '__main__': 

    myDir = '/vision/group/Charades_RGB/Charades_v1_rgb'
    maxFrames = float('-inf')
    for subdir, dirs, files in os.walk(myDir):
        for i, d in enumerate(dirs):
            if i % 100 == 0:
                print('After {}: {}'.format(i, maxFrames))
            maxFrames = max(maxFrames, len(os.listdir(os.path.join(myDir, d))))
        break
    print('Min number of frames = {}'.format(maxFrames))
    
