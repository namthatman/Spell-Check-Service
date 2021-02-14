from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, HttpResponseNotFound, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import Corpus, Singleton, ModelSpellCheck, ModelPlaceHolder
from functools import lru_cache

import data_cleaning as dc
#import feature_engineer as fe
import numpy as np
import tensorflow as tf

import json
import base64

    
@csrf_exempt
def textprocess(request):
    if request.method == 'POST':
        req = request.POST
        body = json.loads(request.body.decode('utf-8'))
        service_type = body["type"]       
        data = body["data"]
        
        # Create/Load model and queue model instance
        queue_model = ModelPlaceHolder()

        if data != "":
            if service_type == 'async': #async
                url_callback = body["urlCallback"]
                return async_service(url_callback, data)
            else:   #sync
                return sync_service(data)
        else:
            return HttpResponse("Wrong parameter")     

    else:
        return HttpResponse("Wrong request method")
    

def sync_service(data):
    data = str(data)
    
    try:
        # Create/Load model and queue model instance
        queue_model = ModelPlaceHolder()
    
        # Get model from queue
        model = queue_model.get_model()
                    
        # Extract cleaned sentences from input
        corpuses = dc.extract_corpus(data)
        
        # Make input, mask, token
        input_ids,attention_mask,token_type_ids = dc.encode_input(test_input=corpuses, MAX_LEN=96, vocab=model._vocab)
                   
        # Predict output
        output = model.predict(input_ids, attention_mask, token_type_ids, batch_size=32)
        output = output[:,0]         
        
        # Make output obj response
        res = []
        for i in range(len(corpuses)):
            if output[i] <= 0.5:
                color = "#1f1f1f"
            else:
                color = "#f5222d"
            res.append({
                'text': corpuses[i],
                'score': str(output[i]),
                'color': color
                })
                
        res_req = {
            'result': res,
            'version': "1.0",
        }
            
        # Return json response
        return HttpResponse(json.dumps(res_req))
    except Exception as err:
        return HttpResponse(err)
    finally:
        # Return model to queue model
        queue_model.put_model(model)


def async_service(url, data):
    data = str(data)
    url_callback = str(url)
    
    try:
        # Create/Load model and queue model instance
        queue_model = ModelPlaceHolder()
        
        # Put request into queue request
        queue_model.worker.put_request(url_callback, data)
        
        return HttpResponse("Request submitted successfully, Result will sent to url callback later.")
    except Exception as err:
        return HttpResponse(err)
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    