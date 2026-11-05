"""FC wire/lifecycle checks without loading models or calling the reviewer."""
import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys

from fastapi.testclient import TestClient

PATH=Path(__file__).parents[1]/'deploy/fc/gateway.py'
spec=importlib.util.spec_from_file_location('router_fc_test',PATH)
gateway=importlib.util.module_from_spec(spec);sys.modules[spec.name]=gateway;spec.loader.exec_module(gateway)

class FakeRuntime:
    def __init__(self):
        self.router=None;self.failed=False;self.route_lock=asyncio.Lock();self.calls=[]
    async def initialize(self):
        def route(text):
            self.calls.append(text)
            return {'skills':['known-skill-id'],'harness':'claude','model':'deepseek'}
        self.router=SimpleNamespace(route=route)

class FCGatewayTests(unittest.TestCase):
    def setUp(self):
        self.context=TestClient(gateway.create_app(FakeRuntime));self.client=self.context.__enter__()
    def tearDown(self):self.context.__exit__(None,None,None)
    def test_no_inference_before_initialize(self):
        self.assertEqual(self.client.get('/health/live').status_code,200)
        self.assertEqual(self.client.get('/health/ready').status_code,503)
        self.assertEqual(self.client.post('/route',json={'request':'research'}).status_code,503)
    def test_exact_original_result_and_one_call(self):
        self.assertEqual(self.client.post('/initialize').status_code,200)
        self.assertEqual(self.client.get('/health/ready').status_code,200)
        response=self.client.post('/route',json={'request':'research'})
        self.assertEqual(response.json(),{'skills':['known-skill-id'],'harness':'claude','model':'deepseek'})
        self.assertEqual(self.client.app.state.runtime.calls,['research'])
    def test_invalid_bodies_never_reach_router(self):
        self.client.post('/initialize')
        for body in [{},{'request':''},{'request':'  '},{'request':3},{'request':'x','extra':True},{'request':'x'*32001}]:
            with self.subTest(body_keys=list(body)):
                self.assertEqual(self.client.post('/route',json=body).status_code,400)
        self.assertEqual(self.client.app.state.runtime.calls,[])
    def test_duplicate_keys_rejected(self):
        self.assertEqual(self.client.post('/route',content='{"request":"a","request":"b"}',headers={'Content-Type':'application/json'}).status_code,400)
    def test_content_type_and_limit(self):
        self.assertEqual(self.client.post('/route',content='hello').status_code,415)
        self.assertEqual(self.client.post('/route',content=b'x'*1048577,headers={'Content-Type':'application/json'}).status_code,413)
    def test_schema_documents_bearer_and_exact_output(self):
        schema=self.client.get('/openapi.json').json()
        self.assertEqual(schema['security'],[{'FCBearer':[]}])
        self.assertEqual(set(schema['components']['schemas']['RouteOutput']['properties']),{'skills','harness','model'})
    def test_review_error_has_no_fabricated_success(self):
        self.client.post('/initialize')
        def fail(_):raise gateway.ReviewError('synthetic upstream failure')
        self.client.app.state.runtime.router.route=fail
        response=self.client.post('/route',json={'request':'research'})
        self.assertEqual(response.status_code,502)
        self.assertNotIn('skills',response.json())

if __name__=='__main__':unittest.main()
