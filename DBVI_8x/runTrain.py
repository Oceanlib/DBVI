import os

jobName = '8x_GoPro'
# jobName = '8x_XVFI'

part = 'Pixel'


freeNodes = ['SH-IDC1-10-5-39-55', 'SH-IDC1-10-5-31-54']
gpuDict = "\"{\'SH-IDC1-10-5-39-55\': \'0,1,2,3,4,5,6,7\', \'SH-IDC1-10-5-31-54\': \'0,1,2,3,4,5,6,7\'}\""


ntaskPerNode = 8  # number of GPUs per nodes
cpus_per_task = 7  # number of CPUs per task
reuseGPU = 1
envDistributed = 1

nodeNum = len(freeNodes)
nTasks = ntaskPerNode * nodeNum if envDistributed else 1
nodeList = ','.join(freeNodes)
initNode = freeNodes[0]

scrip = 'train'
config = 'configTrain'


def runDist():
    pyCode = []
    pyCode.append('python')
    pyCode.append('-m')
    pyCode.append(scrip)
    pyCode.append('--initNode {}'.format(initNode))
    pyCode.append('--config {}'.format(config))
    pyCode.append('--gpuList {}'.format(gpuDict))
    pyCode.append('--reuseGPU {}'.format(reuseGPU))
    pyCode.append('--expName {}'.format(jobName))
    pyCode = ' '.join(pyCode)

    srunCode = []
    srunCode.append('srun')
    srunCode.append('--gres=gpu:{}'.format(ntaskPerNode))
    srunCode.append('--job-name={}'.format(jobName))
    srunCode.append('--partition={}'.format(part))
    srunCode.append('--nodelist={}'.format(nodeList)) if freeNodes is not None else print('Get node by slurm')
    srunCode.append('--ntasks={}'.format(nTasks))
    srunCode.append('--nodes={}'.format(nodeNum))
    srunCode.append(f'--ntasks-per-node={ntaskPerNode}') if envDistributed else print(
        'ntasks-per-node is 1')
    srunCode.append(f'--cpus-per-task={cpus_per_task}')
    srunCode.append('--kill-on-bad-exit=1')
    srunCode.append('--mpi=pmi2')
    srunCode.append(pyCode)

    srunCode = ' '.join(srunCode)
    print(srunCode)

    os.system(srunCode)


if __name__ == '__main__':
    runDist()
