import json
import os

class SettingManager:
    def __init__(self, file='setting.json'):
        self.file = file
        self.setting = self._load_config()
    
    def _load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), self.file)
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"config file {config_path} not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"config file format error: {e}")
    
    def get(self, key: str, default=None):
        return self.setting.get(key, default)
    
    def get_llm_provider(self) -> str:
        return self.get("LLM_PROVIDER", "GEMINI")
    
    def get_gemini_config(self):
        return {
            "api_key": self.get("GEMINI_API_KEY"),
            "model": self.get("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
        }
    
    def get_openai_config(self):
        return {
            "api_key": self.get("OPENAI_API_KEY"),
            "model": self.get("OPENAI_MODEL", "gpt-4o")
        }
    
    def get_deepseek_config(self):
        return {
            "api_key": self.get("DEEPSEEK_API_KEY"),
            "model": self.get("DEEPSEEK_MODEL", "deepseek-chat")
        }
    
    def get_model_paths(self):
        return {
            'gcn': self.setting['GCN_MODEL_PATH'],
            'sentence_transformer': self.setting['SENTENCE_TRANSFORMER_MODEL']
        }
    
    def get_data_paths(self):
        return {
            'example_library': self.setting['EXAMPLE_LIBRARY_PATH'],
            'intent_types': self.setting['INTENT_TYPES_PATH']
        }
    
    def get_input_paths(self):
        return {
            'intent': self.setting['INPUT_INTENT_FILE'],
            'topology': self.setting['INPUT_TOPOLOGY_FILE']
        }
    
    def get_output_paths(self):
        return {
            'responses': self.setting['RESPONSES_DIR'],
            'output': self.setting['OUTPUT_DIR'],
            'configs': self.setting['CONFIGS_DIR']
        }
    
    def get_prompt_paths(self):
        return {
            'intent_process': self.setting['INTENT_PROCESS_FILE'],
            'intent_specification': self.setting['INTENT_SPECIFICATION_FILE'],
            'reflection_prefix': self.setting['REFLECTION_PREFIX_FILE'],
            'interface_suffix': self.setting['INTERFACE_SUFFIX_FILE'],
            'ospf_suffix': self.setting['OSPF_SUFFIX_FILE'],
            'bgp_suffix': self.setting['BGP_SUFFIX_FILE'],
            'bgp_routing_policy': self.setting['BGP_ROUTING_POLICY_FILE'],
            'routemap_format': self.setting['ROUTEMAP_FORMAT_FILE']
        }
    
    def get_intent_types_role_prompt_dir(self):
        return self.setting['INTENT_TYPES_ROLE_PROMPT_DIR']
    
    def get_training_config(self):
        return {
            'batch_size': self.setting['BATCH_SIZE']
        }

# global parameters
setting = SettingManager()