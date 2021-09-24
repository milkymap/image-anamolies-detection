import zmq 
import tqdm 
import json 

import ngtpy 

import numpy as np 
import pickle as pk 

import operator as op 
import itertools as it, functools as ft 

import torch as th 
import torch.nn as nn 
import multiprocessing as mp 

from os import getenv
from libraries.log import logger 
from libraries.strategies import * 

from torch.utils.data import TensorDataset, DataLoader

class ZMQNGT:
    def __init__(self, source, model_path, batch_size, router_port, index, location, dimension):
        self.index = path.join(index, 'ngt_dump')
        self.source = source 
        self.model_path = model_path
        
        self.location = location
        self.dimension = dimension

        self.batch_size = batch_size 
        self.router_port = router_port
        
        self.image_paths = sorted(pull_files(self.source, extension='*.jpg'))
        logger.success(f'nb available images {len(self.image_paths)}')
        
        self.device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
        
        self.resnet18 = th.load(path.join(self.model_path, 'res18.th'), map_location=self.device).eval()
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])

        for prm in self.resnet18.parameters():
            prm.requires_grad = False 
        logger.success(f'the resnet model was loaded on {self.device}')

        if not path.isdir(self.index):
            ngtpy.create(self.index.encode(), dimension=self.dimension)
            self.engine = ngtpy.Index(self.index.encode())
            logger.debug('creation of the index ...!')
            vector_paths = sorted(pull_files(self.location, extension='*.pkl'))
            for v_path in vector_paths:
                logger.debug(f'indexation of the packet {v_path}')
                with open(v_path, 'rb') as fp:
                    loaded_data = np.squeeze(pk.load(fp))
                    tensor_data = th.as_tensor(loaded_data)
                    dataset = TensorDataset(tensor_data)
                    dataloader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size)

                    for tensor_batch in tqdm.tqdm(dataloader):
                        self.engine.batch_insert(th.vstack(tensor_batch).numpy())
            self.engine.save()
        else:
            self.engine = ngtpy.Index(self.index.encode())   
        # end if ...! 
         
    def start(self):
        ZMQ_INITIALIZED = False 
        try:
            ctx = zmq.Context() 
            router_socket = ctx.socket(zmq.ROUTER)
            router_socket.bind(f'tcp://*:{self.router_port}')
            router_controller = zmq.Poller()
            router_controller.register(router_socket, zmq.POLLIN)

            ZMQ_INITIALIZED = True 
            logger.success('router server is initialized 100%')

            req_memory, rep_memory = mp.Queue(), mp.Queue()
            keep_routing = True 
            while keep_routing:
                incoming_events = dict(router_controller.poll(100))
                logger.debug(f'router is listening on port {self.router_port}')

                if router_socket in incoming_events:
                    if incoming_events[router_socket] == zmq.POLLIN:
                        remote_address, _, path2image = router_socket.recv_multipart()
                        req_memory.put((remote_address, path2image.decode()))
                # end if ...! 

                if not req_memory.empty():
                    target_address, target_path = req_memory.get()
                    _, image_filename = path.split(target_path)
                    image_local_path = path.join(self.source, image_filename)

                    if path.isfile(image_local_path):
                        try:
                            cv_image = read_image(image_local_path)
                            th_image = cv2th(cv_image)
                            prepared_image = prepare_image(th_image).to(self.device)
                            extracted_features = np.squeeze(self.resnet18(prepared_image[None, ...]).numpy())
                            logger.debug(extracted_features.shape)
                            res_search = self.engine.search(extracted_features, 7)
                            print(res_search)
                            positions, distances = list(zip(*res_search))
                            selected_candidates = op.itemgetter(*positions)(self.image_paths)
                            print(selected_candidates)

                            detected_candidates = search_duplication(image_local_path, selected_candidates, (64, 64), 0.5, 0.75)
                            print(detected_candidates)
                            
                            rep_memory.put((target_address, detected_candidates))
                        except Exception as e:
                            logger.warning(e)
                            
                    else:
                        logger.error(f'{image_local_path} is not a valid path ...!')
                        rep_memory.put((target_address, []))
                # end if 

                if not rep_memory.empty():
                    address_to_send, response_to_send = rep_memory.get()
                    packaged_response = build_response(response_to_send)
                    
                    router_socket.send_multipart(
                        [
                            address_to_send, 
                            b'', 
                            json.dumps(packaged_response).encode()
                        ]
                    )
                # end if
                  
            # end routing loop 
        except KeyboardInterrupt as e:
            logger.warning(e)
        except Exception as e: 
            logger.error(e)
        finally:
            if ZMQ_INITIALIZED:
                router_controller.unregister(router_socket)
                router_socket.close()
                ctx.term()
            logger.success('zmq ressources free  100%')

def main():
    INDEX = getenv('INDEX')
    SOURCE = getenv('SOURCE')
    LOCATION = getenv('LOCATION')
    MODEL_PATH = getenv('MODEL_PATH')
    
    DIMENSION = int(getenv('DIMENSION'))
    BATCH_SIZE = int(getenv('BATCH_SIZE'))
    ROUTER_PORT = int(getenv('ROUTER_PORT'))

    server = ZMQNGT(SOURCE, MODEL_PATH, BATCH_SIZE, ROUTER_PORT, INDEX, LOCATION, DIMENSION)
    server.start()

if __name__ == '__main__':
    logger.debug(' ... [NGT server] ... ')
    main()
    