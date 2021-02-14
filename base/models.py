import queue, threading, urllib
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound, JsonResponse
from django.db import models
from fairseq.data import Dictionary
import tensorflow as tf
import data_cleaning as dc
import os
    
    
class Corpus(models.Model):
    text_id = models.IntegerField(primary_key=True)
    text = models.CharField(max_length=512)
    text_class = models.IntegerField(default=0)
    
    def __str__(self):
        return "[%s] %s" % (self.text_id, self.text, self.text_class)
    
    
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    

class ModelSpellCheck(metaclass=Singleton):
    def __init__(self):
        #tf.debugging.set_log_device_placement(True)
        #config = tf.compat.v1.ConfigProto()
        #config.gpu_options.allow_growth=True
        #sess = tf.compat.v1.Session(config=config)
        #gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.233)       
        #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2560)])
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        
        self._model = tf.keras.models.load_model("resources/saved_model/gpu_phobert")
        
        self._vocab = Dictionary()
        self._vocab.add_from_file("vocab/new_vocab.txt")
        
        
    def predict(self, input_ids, attention_mask, token_type_ids, batch_size):
        misspell_predicted = self._model.predict([input_ids,attention_mask,token_type_ids], batch_size=batch_size)
        return misspell_predicted
    

class ModelPlaceHolder(metaclass=Singleton):
    def __init__(self):
        self.queue_model = queue.Queue(maxsize=1)
        model = ModelSpellCheck()
        self.queue_model.put(model, block=True, timeout=None)
        
        self.worker = Worker()
        self.worker.start()
    
    def get_model(self):
        return self.queue_model.get(block=True, timeout=None)
    
    def put_model(self, model):
        self.queue_model.put(model, block=True, timeout=None)
        
        
class Worker(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.queue_request = queue.Queue(maxsize=100)        
    
    def put_request(self, url, data):
        try:
            self.queue_request.put({
                'url_callback': url,
                'data': data
                }, block=False, timeout=None)
        except queue.Full as err:
            return HttpResponse("Worker: Queue full")
        except Exception as err:
            return HttpResponse("Worker: Queue failed")
       
    def run(self):
        while True:
            if self.queue_request.empty() is False: 
                try:
                    print("Worker started")
                    
                    # Create/Load model and queue model instance
                    queue_model = ModelPlaceHolder()                    
                    # Get model from queue
                    model = queue_model.get_model()
                    
                    # Get request from queue request
                    req = self.queue_request.get()
                    url_callback = req['url_callback']
                    data = req['data']
                    
                    # Extract cleaned sentences from input
                    corpuses = dc.extract_corpus(data)           
                    
                    # Make input, mask, token
                    input_ids,attention_mask,token_type_ids = dc.encode_input(test_input=corpuses, MAX_LEN=96, vocab=model._vocab)
                               
                    # Predict output
                    output = model.predict(input_ids, attention_mask, token_type_ids, batch_size=32)
                    output = output[:,0]     
                    
                    # Make output obj response
                    body = []
                    for i in range(len(corpuses)):
                        body.append({
                            'text': corpuses[i],
                            'class': str(output[i])
                            })
                        
                    req = urllib.request.Request(
                        url_callback,
                        data=body,
                        headers={'content-type': 'application/json'},
                        )   
                    
                    # Send json response to url callback
                    res = urllib.request.urlopen(req)
                    
                    print("Worker finished")
                    
                except Exception as err:
                    print(err)
                finally:
                    # Return model to queue model
                    queue_model.put_model(model)


                    
        