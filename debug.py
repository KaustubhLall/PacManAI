import tensorflow as tf
import sys

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")

# Check for CUDA support
print(f"Is built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"Is GPU available: {tf.test.is_gpu_available()}")

# Get the list of GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {gpus}")

# Get CUDA and cuDNN version if available
cuda_version = tf.sysconfig.get_build_info().get('cuda_version')
cudnn_version = tf.sysconfig.get_build_info().get('cudnn_version')

print(f"CUDA version: {cuda_version}")
print(f"CUDNN version: {cudnn_version}")

# If CUDA is available, try to create a simple TensorFlow operation and execute it on GPU
if tf.test.is_built_with_cuda():
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
        c = tf.matmul(a, b)

    print(f"Result of a TensorFlow GPU operation: {c}")
