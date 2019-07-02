import svhn2mnist
import usps
import syn2gtrsb
# import syndig2svhn

def Generator(source, target, pixelda=False):
    # if source == 'usps' or target == 'usps':
    #     return usps.Feature()
    # else:
    return svhn2mnist.Feature_base()
    # elif source == 'synth':
    #     return syn2gtrsb.Feature()

def Disentangler():
    return svhn2mnist.Feature_disentangle()

def Classifier(source, target):
    # if source == 'usps' or target == 'usps':
        # return usps.Predictor()
    # else:
    return svhn2mnist.Predictor()
    # if source == 'synth':
    #     return syn2gtrsb.Predictor()
def Feature_Discriminator():
    return svhn2mnist.Feature_discriminator()

def Reconstructor():
    return svhn2mnist.Reconstructor()

def Mine():
    return svhn2mnist.Mine()
