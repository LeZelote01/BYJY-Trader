#!/usr/bin/env python3
"""
Debug script for optimization endpoints
"""

import asyncio
import aiohttp
import json

BACKEND_URL = "https://e61aade5-33d8-44a4-8cd5-e207b55d780e.preview.emergentagent.com"

async def debug_optimization():
    async with aiohttp.ClientSession() as session:
        print("🔍 Testing optimization endpoints...")
        
        # 1. Test service status
        print("\n1. Testing service status...")
        async with session.get(f"{BACKEND_URL}/api/optimization/status") as resp:
            print(f"Status: {resp.status}")
            data = await resp.json()
            print(f"Response: {json.dumps(data, indent=2)}")
        
        # 2. Start a genetic optimization
        print("\n2. Starting genetic optimization...")
        request_data = {
            "parameter_space": {
                "parameters": {
                    "layers": {"type": "int", "min": 1, "max": 2},
                    "neurons": {"type": "int", "min": 32, "max": 64}
                }
            },
            "optimization_config": {
                "population_size": 10,
                "num_generations": 10
            },
            "target_model": "lstm"
        }
        
        async with session.post(
            f"{BACKEND_URL}/api/optimization/genetic/start",
            json=request_data,
            headers={'Content-Type': 'application/json'}
        ) as resp:
            print(f"Status: {resp.status}")
            data = await resp.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            
            if resp.status == 200 and 'job_id' in data:
                job_id = data['job_id']
                
                # 3. Test status with valid job ID
                print(f"\n3. Testing status with job ID: {job_id}")
                await asyncio.sleep(1)  # Wait a bit
                
                async with session.get(f"{BACKEND_URL}/api/optimization/genetic/status/{job_id}") as status_resp:
                    print(f"Status: {status_resp.status}")
                    if status_resp.status == 200:
                        status_data = await status_resp.json()
                        print(f"Response: {json.dumps(status_data, indent=2)}")
                    else:
                        error_data = await status_resp.text()
                        print(f"Error: {error_data}")

if __name__ == "__main__":
    asyncio.run(debug_optimization())