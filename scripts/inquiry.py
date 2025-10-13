import pandas as pd                                                                                                                                                                   
                                                                                                                                                                                        
# Read the CSV file                                                                                                                                                                   
df = pd.read_csv('/fs/nexus-projects/sim2real/aliu/navsim/exp/transfuser_agent_subset/2025.10.12.10.56.51/2025.10.12.11.06.04.csv')                                                   
                                                                                                                                                                                        
# Filter to only valid scenarios (exclude the summary rows)                                                                                                                           
valid_scenarios = df[df['valid'] == True].copy()                                                                                                                                      
                                                                                                                                                                                        
print(f"Total scenarios evaluated: {len(valid_scenarios)}")                                                                                                                           
print(f"\n{'='*60}")                                                                                                                                                                  
print(f"OVERALL DRIVING SCORES")                                                                                                                                                      
print(f"{'='*60}\n")                                                                                                                                                                  
                                                                                                                                                                                        
# Overall score statistics                                                                                                                                                            
print(f"Mean Score:   {valid_scenarios['score'].mean():.4f}")                                                                                                                         
print(f"Median Score: {valid_scenarios['score'].median():.4f}")                                                                                                                       
print(f"Min Score:    {valid_scenarios['score'].min():.4f}")                                                                                                                          
print(f"Max Score:    {valid_scenarios['score'].max():.4f}")                                                                                                                          
print(f"Std Dev:      {valid_scenarios['score'].std():.4f}")                                                                                                                          
                                                                                                                                                                                        
print(f"\n{'='*60}")                                                                                                                                                                  
print(f"INDIVIDUAL METRIC SCORES (Stage One)")                                                                                                                                        
print(f"{'='*60}\n")                                                                                                                                                                  
                                                                                                                                                                                        
# Individual metrics                                                                                                                                                                  
metrics = {                                                                                                                                                                           
    'No At-Fault Collisions': 'no_at_fault_collisions_stage_one',                                                                                                                     
    'Drivable Area Compliance': 'drivable_area_compliance_stage_one',                                                                                                                 
    'Driving Direction Compliance': 'driving_direction_compliance_stage_one',                                                                                                         
    'Traffic Light Compliance': 'traffic_light_compliance_stage_one',                                                                                                                 
    'Ego Progress': 'ego_progress_stage_one',                                                                                                                                         
    'Time to Collision Within Bound': 'time_to_collision_within_bound_stage_one',                                                                                                     
    'Lane Keeping': 'lane_keeping_stage_one',                                                                                                                                         
    'History Comfort': 'history_comfort_stage_one'                                                                                                                                    
}                                                                                                                                                                                     
                                                                                                                                                                                        
for name, col in metrics.items():                                                                                                                                                     
    mean_val = valid_scenarios[col].mean()                                                                                                                                            
    print(f"{name:.<45} {mean_val:.4f}")                                                                                                                                              
                                                                                                                                                                                        
# Count perfect scores                                                                                                                                                                
perfect_scores = (valid_scenarios['score'] == 1.0).sum()                                                                                                                              
print(f"\nScenarios with perfect score (1.0): {perfect_scores} / {len(valid_scenarios)} ({100*perfect_scores/len(valid_scenarios):.1f}%)")                                            
                                                                                                                                                                                        
# Count scenarios with score > 0.9                                                                                                                                                    
high_scores = (valid_scenarios['score'] > 0.9).sum()                                                                                                                                  
print(f"Scenarios with score > 0.9:         {high_scores} / {len(valid_scenarios)} ({100*high_scores/len(valid_scenarios):.1f}%)")                                                    
                                                                                                                                                                                        
# Count scenarios with score == 0                                                                                                                                                     
zero_scores = (valid_scenarios['score'] == 0.0).sum()                                                                                                                                 
if zero_scores > 0:                                                                                                                                                                   
    print(f"\nWARNING: {zero_scores} scenarios have score = 0.0")      