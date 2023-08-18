from valve_overlapping_genetic_algo import *
import streamlit as st
import pandas as pd
from PIL import Image
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
from matplotlib.font_manager import FontProperties
from plotly.tools import mpl_to_plotly
import plotly.graph_objects as go


st.set_page_config(page_title="Well Scheduling", page_icon=":alarm_clock:")

def home():
    img_glv = Image.open("images/M1.png")
    
    st.write("# Optimization of Time Cycle Staggering for Intermittent Gas Lift Wells Using Genetic Algorithm")
    st.write("---")
    image_column, text_column = st.columns((1, 2))
    with image_column:
        st.image(img_glv)
    with text_column:
        st.write("## Introduction:")
        text="Intermittent gas lift wells play a crucial role in oil and gas production, but the simultaneous gas injection ON time across multiple wells connected to a common gas source can lead to undesirable consequences. Fluctuations in gas injection header pressure and liquid surges in well fluid header can reduce the efficiency of gas lift operations. Manual staggering of time cycles has been the prevailing practice, but it becomes increasingly challenging with a larger number of wells and time slots."
        st.markdown(f"<p style='text-align: justify;'>{text}</p>", unsafe_allow_html=True)
        st.write("---")
    image_column, text_column = st.columns((1, 2))
    img_genalg = Image.open("images/genalg.jpg")
    with image_column:
        st.image(img_genalg)
    with text_column:
        st.write("## Genetic Algorithm Approach:")
        text="Genetic algorithms provide an effective optimization technique for addressing the time cycle staggering in intermittent gas lift wells. The algorithm involves creating a population of potential solutions, representing each solution as a set of chromosomes or genes. In the context of this model, the gas injection time slots for each well are encoded as chromosomes. The objective function evaluates the level of interference between the gas injection timings of different wells."
        st.markdown(f"<p style='text-align: justify;'>{text}</p>", unsafe_allow_html=True)
        st.write("[Learn More>](https://www.geeksforgeeks.org/genetic-algorithms/)")
        st.write("---")
    image_column, text_column = st.columns((1, 2))
    img_genalg = Image.open("images/nphard.jpg")
    with image_column:
        st.image(img_genalg)
    with text_column:
        st.write("## Dealing with NP-Hardness:")
        text="Time cycle staggering for intermittent gas lift wells falls into the class of NP-hard problems, meaning finding the exact optimal solution within a reasonable timeframe is computationally infeasible. However, genetic algorithms provide a heuristic approach to tackle this challenge efficiently. By leveraging the principles of natural selection and evolution, genetic algorithms explore a vast search space and iteratively improve the solutions until convergence."
        st.markdown(f"<p style='text-align: justify;'>{text}</p>", unsafe_allow_html=True)
        st.write("[Learn More>](https://www.geeksforgeeks.org/types-of-complexity-classes-p-np-conp-np-hard-and-np-complete/)")
    st.write("---")
    st.write("## Conclusion:")
    text="The development of a mathematical model based on a genetic algorithm optimization approach offers an automated and efficient solution for time cycle staggering in intermittent gas lift wells. By minimizing gas injection interference, this optimization technique enhances the overall efficiency of gas lift operations, preventing production losses. Despite the NP-hard nature of the problem, genetic algorithms provide an effective means of generating near-optimal solutions within a reasonable computational time. This advancement in optimization methodology holds great promise for the oil and gas industry, facilitating the optimization of gas injection time cycles and improving production efficiency."
    st.markdown(f"<p style='text-align: justify;'>{text}</p>", unsafe_allow_html=True)
    st.write("---")
    #st.image("image1.jpg", caption="Image 1")
    #st.image("image2.jpg", caption="Image 2")
    

def analysis():
    
    st.sidebar.header("Data Analysis")
    num_data_points = st.sidebar.number_input("Number of wells", min_value=1, max_value=100, value=2)
    iv=st.sidebar.checkbox("Interactive View")

    
    
    data_table = pd.DataFrame(index=range(num_data_points), columns=['Well No', 'Gas Injection OFF time, min', 'Gas Injection ON time, min','Priority'])
    
    grid_options = {
        'columnDefs': [
            {'headerName': 'Well No.', 'field': 'Well No', 'width': 150, 'editable': True},
            {'headerName': 'Gas Injection OFF time, min', 'field': 'Gas Injection OFF time, min', 'width': 200, 'editable': True},
            {'headerName': 'Gas Injection ON time, min', 'field': 'Gas Injection ON time, min', 'width': 200, 'editable': True},
            {'headerName': 'Priority', 'field': 'Priority', 'width': 150, 'editable': True},
        ],
        'headerHeight': 50,  # Adjust the header height
        'floatingFilter': True,  # Enable floating filter
        'width': 50  # Adjust the width of the entire table
    }

    st.markdown('# Enter input data below:')
    st.write("### Instruction:")
    st.write("- In priority column, if you want to give priority to a particular well then enter 1 else 0.")
    st.write("- It might take a couple of minutes to generate the results. Please wait.")
    grid_table = AgGrid(data_table,gridOptions=grid_options)
        
    
    with st.container():
        _,btn,_=st.columns((2,1,2))
        with btn:
            calculate_button = st.button('Calculate')
    try:
        if calculate_button:
            #st.write(iv)
            #st.write(data_table.columns)
            data_table=pd.DataFrame(grid_table['data'])
            data_table["Well No"]=data_table["Well No"].astype(str)
            data_table["Gas Injection OFF time, min"]=data_table["Gas Injection OFF time, min"].astype(int)
            data_table["Gas Injection ON time, min"]=data_table["Gas Injection ON time, min"].astype(int)            
            data_table.replace("None", 0, inplace=True)
            data_table["Priority"]=data_table["Priority"].astype(int)
            
            data_tuples = data_table.values.T.tolist()
            

            input_valves=[]

            for i,j,k in zip(data_tuples[1],data_tuples[2],data_tuples[3]):
                input_valves.append((i,j,k))
            
            #input_valves=[(55,5,1),(45,15,1),(55,5,0),(230,10,1),(55,5,0),(55,5,0),(50,10,0),(45,15,0),(50,10,0)]
            normal_calculation=True

            if normal_calculation==False:
                population_size=500
                gen_theshold=100
            else:
                population_size=200
                gen_theshold=20

            schedule,sum_timings,overlap_number,priority_number=valve_overlapping(input_valves,population_size=population_size,gen_theshold=gen_theshold)
            #st.write(overlap_number)
            sum_timings=list(itertools.chain.from_iterable(sum_timings))
            #print("Maximum Overlap Value Count:",overlap_number)
            #print("Priority Value:",priority_number)
            #print("Valve schedule",schedule)
            #st.write(sum_timings)
            time=np.arange(0, len(sum_timings))
            valve_numbers=sum_timings

            # Plot the orthogonal step function
            fig,ax=plt.subplots()

            for i in range(len(time) - 1):
                ax.plot([time[i], time[i + 1]], [valve_numbers[i], valve_numbers[i]], color='blue')
                ax.plot([time[i + 1], time[i + 1]], [valve_numbers[i], valve_numbers[i + 1]], color='blue')
                x = [time[i], time[i + 1], time[i + 1], time[i]]
                y = [valve_numbers[i], valve_numbers[i], 0, 0]
                ax.fill(x, y, color='lightblue', alpha=0.5)
            #ax.plot(sum_timings[:240], linestyle='-', color='black')
            arrow_labels=data_table["Well No"].to_list()



            font_properties = FontProperties(family='serif', size=15, weight='normal', style='italic')
            
                
            for value in schedule:
                ax.arrow(value,-0.2, 0, 0.2, head_width=2.5, head_length=0.1, fc='black', ec='black')
            
            #st.write(max(sum_timings))
            #st.write("run")
            y_ticks = range(0, int(max(sum_timings))+3,1)
            #x_ticks = range(0, min(len(sum_timings),241)+20,10)
            plt.yticks(y_ticks)
            #plt.xticks(x_ticks)
            #num_ticks = 20
            #x_ticks = plt.xticks()[0]
            #x_min, x_max = min(x_ticks), max(x_ticks)
            #tick_positions = np.linspace(x_min, x_max, num_ticks, endpoint=True, dtype=int)

            # Apply the tick positions to the plot
            #plt.xticks(tick_positions)

            # Set the x-axis limit to start from zero
            #plt.xlim(left=0)

            #ax.set_title('Indicative valve overlapping curve')
            ax.set_xlabel('Time in minute')
            ax.set_ylabel('Well overlap count')
            ax.grid(color='lightgray', linestyle='--')
            #plt.show()
            
            st.write("---")
            st.write("## Indicative well overlapping diagram:")
            st.pyplot(fig)
            plotly_fig = mpl_to_plotly(fig)
            if iv: 
                st.write("---")
                st.write("## Interactive View:")
                st.plotly_chart(plotly_fig)

            def schedule_to_time(schedule):
                timings=[]
                for i in schedule:
                    partA=str(9+i//60)+str(":")
                    partB=str(i%60) if i%60>9 else str("0")+str(i%60)
                    timings.append(partA+partB)
                return timings
            timings=schedule_to_time(schedule)

            #print(timings)
            #print(data_table)
            data_table["Cycle 1, ON"]=timings
            data_table['Cycle 1, OFF']=schedule_to_time(schedule+data_table["Gas Injection ON time, min"])
            data_table['Cycle 2, ON']=schedule_to_time(schedule+data_table["Gas Injection ON time, min"]+data_table["Gas Injection OFF time, min"])
            data_table['Cycle 2, OFF']=schedule_to_time(schedule+data_table["Gas Injection ON time, min"]+data_table["Gas Injection OFF time, min"]+data_table["Gas Injection ON time, min"])
            #st.write(data_table.columns)
            data_table.rename(columns={'Gas Injection ON time, min': 'GI ON time', 'Gas Injection OFF time, min': 'GI OFF time'}, inplace=True)
            data_table=data_table.set_index('Well No')
            #data_table.reset_index(drop=True, inplace=False)

            #print(data_table)
            
            st.write("---")
            #st.write("---")
            st.write('## No. of gas injection wells VS Time utilization:')
            fig1,ax1=plt.subplots(figsize=(10, 10))
            sum_timings_set=list(set(int(x) for x in sum_timings))
            sum_timings_frequency=[]
            for i in sum_timings_set:
                sum_timings_frequency.append(math.ceil(sum_timings.count(i)/len(sum_timings)*100*100)/100)
            
            mean=sum(sum_timings)/len(sum_timings)
            
            # Plot the data on the new axis
            ax1.plot(sum_timings_set, sum_timings_frequency)

            # Plot the mean value as a dotted vertical line
            ax1.axvline(x=mean, linestyle='dotted', color='red', label='Mean is '+str(round(mean,2)))
            x_ticks=sum_timings_set
            #st.write(x_ticks)
            ax1.set_xticks(x_ticks)
            ax1.legend()
            ax1.grid(color='lightgray', linestyle='--')
            # Set labels for the axes and title
            ax1.set_xlabel('No. of gas injection wells')
            ax1.set_ylabel('Timing percentage of the day (%)')
            #ax1.set_title('Continuous Straight Line Plot')

            # Show the plot
            left_col,centre_col,right_col=st.columns((0.2,1,0.3))
            with centre_col:
                st.pyplot(fig1)
            

            st.write("---")

            st.write("## Scheduling table for wells:")
            st.write("- 9:00 AM is taken as reference time")
            with st.container():
                _,table=st.columns((0.1,2))
                with table:
                    st.write(data_table)

            st.write("---")
            st.write('## Results:')
            overlap_value=int(max(sum_timings)) if max(sum_timings)>1 else "No Overlapping!"
            st.write("- Maximum well overlap: "+str(overlap_value))
            #st.write("- Maximum valve overlap count (in minutes) in a cycle: "+str(overlap_number))
            if priority_number==0:
                st.write("- For priority wells overlapping with other wells has been avoided successfully")
            else:
                st.write("- Well overlapping is present for priority wells, however, minimum overlapping scheduling is done.")

            st.write("---")
            ###########HERE 
    except ValueError: 
        #st.write(ValueError)
        st.warning("Data filled incorrectly! Please check again :)")

def cluster_wise_analysis():
    
    st.sidebar.header("Data Analysis")
    num_data_points = st.sidebar.number_input("Number of wells", min_value=1, max_value=100, value=2)
    iv=st.sidebar.checkbox("Interactive View")

    
    
    data_table = pd.DataFrame(index=range(num_data_points), columns=['Well No', 'Gas Injection OFF time, min', 'Gas Injection ON time, min','Priority'])
    
    grid_options = {
        'columnDefs': [
            {'headerName': 'Well No.', 'field': 'Well No', 'width': 150, 'editable': True},
            {'headerName': 'Gas Injection OFF time, min', 'field': 'Gas Injection OFF time, min', 'width': 200, 'editable': True},
            {'headerName': 'Gas Injection ON time, min', 'field': 'Gas Injection ON time, min', 'width': 200, 'editable': True},
            {'headerName': 'Cluster Name', 'field': 'Priority', 'width': 150, 'editable': True},
        ],
        'headerHeight': 50,  # Adjust the header height
        'floatingFilter': True,  # Enable floating filter
        'width': 50  # Adjust the width of the entire table
    }

    st.markdown('# Enter input data below:')
    st.write("### Instruction:")
    st.write("- In cluster name column, give name of cluster of particular well")
    st.write("- It might take a couple of minutes to generate the results. Please wait.")
    grid_table = AgGrid(data_table,gridOptions=grid_options)
        
    
    with st.container():
        _,btn,_=st.columns((2,1,2))
        with btn:
            calculate_button = st.button('Calculate')
    try:
        if calculate_button:
            #st.write(iv)
            #st.write(data_table.columns)
            data_table=pd.DataFrame(grid_table['data'])
            data_table["Well No"]=data_table["Well No"].astype(str)
            data_table["Gas Injection OFF time, min"]=data_table["Gas Injection OFF time, min"].astype(int)
            data_table["Gas Injection ON time, min"]=data_table["Gas Injection ON time, min"].astype(int)            
            data_table.replace("None", 0, inplace=True)
            data_table["Priority"]=data_table["Priority"].astype(str)
            
            grouped_data_table=data_table.groupby('Priority')

            cycles=[]
            cluster_sum_timings=[]
            cluster_schedule=[]
            cluster_priority=[]
            for cluster_name, cluster_data in grouped_data_table:
                
                cluster_data_without_cluster = cluster_data.drop(columns=['Priority'])
                data_table=cluster_data_without_cluster



                data_tuples = data_table.values.T.tolist()
            

                input_valves=[]

                for i,j,k in zip(data_tuples[1],data_tuples[2],[0]*len(data_tuples[2])):
                    input_valves.append((i,j,k))
                
                #input_valves=[(55,5,1),(45,15,1),(55,5,0),(230,10,1),(55,5,0),(55,5,0),(50,10,0),(45,15,0),(50,10,0)]
                normal_calculation=True

                if normal_calculation==False:
                    population_size=500
                    gen_theshold=100
                else:
                    population_size=200
                    gen_theshold=20

                schedule,sum_timings,overlap_number,priority_number=valve_overlapping(input_valves,population_size=population_size,gen_theshold=gen_theshold)
                
                #st.write(overlap_number)
                sum_timings=list(itertools.chain.from_iterable(sum_timings))

                cluster_schedule.append(schedule)
                cluster_sum_timings.append(sum_timings)
                cluster_priority.append(priority_number)

                cycles.append(tuple(sum_timings))
                
            
            len_cycles=[len(i) for i in cycles]
            LCM_cycles=math.lcm(*len_cycles)
            final_cycles=[cycle*int(LCM_cycles/len(cycle)) for cycle in cycles]

            overall_schedule,overall_sum_timings,overlap_type,priority_number=valve_overlapping(cycles=final_cycles)
            #st.write("# Original overlap type")
            #st.write(overlap_type)
            shift_factor=[x-min(overall_schedule) for x in overall_schedule]
            
            #st.write(shift_factor)
            #######For plotting with modifications 
            overall_cluster_schedule=[]
            overall_cluster_sum_timings=[]
            loop_count=-1
            for cluster_name, cluster_data in grouped_data_table:
                loop_count+=1
                cluster_data_without_cluster = cluster_data.drop(columns=['Priority'])
                data_table=cluster_data_without_cluster



                data_tuples = data_table.values.T.tolist()
            

                input_valves=[]

                for i,j,k in zip(data_tuples[1],data_tuples[2],[0]*len(data_tuples[2])):
                    input_valves.append((i,j,k))
                
                #input_valves=[(55,5,1),(45,15,1),(55,5,0),(230,10,1),(55,5,0),(55,5,0),(50,10,0),(45,15,0),(50,10,0)]
                normal_calculation=True

                if normal_calculation==False:
                    population_size=500
                    gen_theshold=100
                else:
                    population_size=200
                    gen_theshold=20

                schedule,sum_timings,priority_number=cluster_schedule[loop_count],cluster_sum_timings[loop_count],cluster_priority[loop_count]
                #st.write(overlap_number)
                #st.write(sum_timings)
                #sum_timings=list(itertools.chain.from_iterable(sum_timings))
                #schedule=schedule[-shift_factor[loop_count]:]+schedule[:-shift_factor[loop_count]]
                ########WORKING HERE
                
                #st.write(schedule)
                #st.write(sum_timings)
                schedule=[x+shift_factor[loop_count] for x in schedule]
                schedule=[x if x<=len(sum_timings) else x%len(sum_timings) for x in schedule]

                #schedule=schedule_final
                sum_timings=sum_timings[-shift_factor[loop_count]%len(sum_timings):]+sum_timings[:-shift_factor[loop_count]%len(sum_timings)]
                
                overall_cluster_schedule.append(schedule)
                overall_cluster_sum_timings.append(sum_timings)
                
                
                #cycles.append(tuple(sum_timings))
                #print("Maximum Overlap Value Count:",overlap_number)
                #print("Priority Value:",priority_number)
                #print("Valve schedule",schedule)
                #st.write(sum_timings)
                time=np.arange(0, len(sum_timings))
                valve_numbers=sum_timings

                # Plot the orthogonal step function
                fig,ax=plt.subplots()

                for i in range(len(time) - 1):
                    ax.plot([time[i], time[i + 1]], [valve_numbers[i], valve_numbers[i]], color='blue')
                    ax.plot([time[i + 1], time[i + 1]], [valve_numbers[i], valve_numbers[i + 1]], color='blue')
                    x = [time[i], time[i + 1], time[i + 1], time[i]]
                    y = [valve_numbers[i], valve_numbers[i], 0, 0]
                    ax.fill(x, y, color='lightblue', alpha=0.5)
                #ax.plot(sum_timings[:240], linestyle='-', color='black')
                arrow_labels=data_table["Well No"].to_list()



                font_properties = FontProperties(family='serif', size=15, weight='normal', style='italic')
                

                    
                for value in schedule:
                    ax.arrow(value,-0.2, 0, 0.2, head_width=2.5, head_length=0.1, fc='black', ec='black')
                
                #st.write(max(sum_timings))
                #st.write("run")
                y_ticks = range(0, int(max(sum_timings))+3,1)
                #x_ticks = range(0, min(len(sum_timings),241)+20,10)
                plt.yticks(y_ticks)
                #plt.xticks(x_ticks)
                #num_ticks = 20
                #x_ticks = plt.xticks()[0]
                #x_min, x_max = min(x_ticks), max(x_ticks)
                #tick_positions = np.linspace(x_min, x_max, num_ticks, endpoint=True, dtype=int)

                # Apply the tick positions to the plot
                #plt.xticks(tick_positions)

                # Set the x-axis limit to start from zero
                #plt.xlim(left=0)

                #ax.set_title('Indicative valve overlapping curve')
                ax.set_xlabel('Time in minute')
                ax.set_ylabel('Well overlap count')
                ax.grid(color='lightgray', linestyle='--')
                #plt.show()
                
                st.write("---")
                st.write("## Indicative well overlapping for cluster "+cluster_name+":")
                st.pyplot(fig)
                plotly_fig = mpl_to_plotly(fig)
                if iv: 
                    st.write("---")
                    st.write("## Interactive View:")
                    st.plotly_chart(plotly_fig)

                def schedule_to_time(schedule):
                    timings=[]
                    for i in schedule:
                        partA=str(9+i//60)+str(":")
                        partB=str(i%60) if i%60>9 else str("0")+str(i%60)
                        timings.append(partA+partB)
                    return timings
                timings=schedule_to_time(schedule)

                #print(timings)
                #print(data_table)
                data_table["Cycle 1, ON"]=timings
                data_table['Cycle 1, OFF']=schedule_to_time(schedule+data_table["Gas Injection ON time, min"])
                data_table['Cycle 2, ON']=schedule_to_time(schedule+data_table["Gas Injection ON time, min"]+data_table["Gas Injection OFF time, min"])
                data_table['Cycle 2, OFF']=schedule_to_time(schedule+data_table["Gas Injection ON time, min"]+data_table["Gas Injection OFF time, min"]+data_table["Gas Injection ON time, min"])
                #st.write(data_table.columns)
                data_table.rename(columns={'Gas Injection ON time, min': 'GI ON time', 'Gas Injection OFF time, min': 'GI OFF time'}, inplace=True)
                data_table=data_table.set_index('Well No')
                #data_table.reset_index(drop=True, inplace=False)
                st.write("---")
                #st.write("---")
                st.write("## Time utilization for cluster "+cluster_name+":")
                fig1,ax1=plt.subplots(figsize=(10, 10))
                sum_timings_set=list(set(int(x) for x in sum_timings))
                sum_timings_frequency=[]
                for i in sum_timings_set:
                    sum_timings_frequency.append(math.ceil(sum_timings.count(i)/len(sum_timings)*100*100)/100)
                
                mean=sum(sum_timings)/len(sum_timings)
                
                # Plot the data on the new axis
                ax1.plot(sum_timings_set, sum_timings_frequency)

                #annotations = [f'({xi}, {yi})' for xi, yi in zip(sum_timings_set, sum_timings_frequency)]
                #for annotation, xy in zip(annotations, zip(x, y)):
                #    ax1.annotate(annotation, xy)
                #    
                #    # Connect dotted orthogonal lines to the x and y axes
                #    ax.plot([xy[0], xy[1]], [xy[1], 0], linestyle='dotted', color='gray')
                #    ax.plot([xy[0], 0], [xy[0], xy[1]], linestyle='dotted', color='gray')

                # Plot the mean value as a dotted vertical line
                ax1.axvline(x=mean, linestyle='dotted', color='red', label='Mean is '+str(round(mean,2)))
                x_ticks=sum_timings_set
                #st.write(x_ticks)
                ax1.set_xticks(x_ticks)
                ax1.legend()
                ax1.grid(color='lightgray', linestyle='--')
                # Set labels for the axes and title
                ax1.set_xlabel('No. of gas injection wells')
                ax1.set_ylabel('Timing percentage of the day (%)')
                #ax1.set_title('Continuous Straight Line Plot')
                #print(data_table)
                    # Show the plot
                left_col,centre_col,right_col=st.columns((0.2,1,0.3))
                with centre_col:
                    st.pyplot(fig1)
            
                
                    

                st.write("---")

                st.write("## Scheduling table for wells for cluster "+cluster_name+":")
                st.write("- 9:00 AM is taken as reference time")
                with st.container():
                    _,table=st.columns((0.1,2))
                    with table:
                        st.write(data_table)
            
            
            
            
            
                #st.write(overlap_number)
            #########WORKING HERE with overall_cluster variables, make them even using LCM 
            schedule=[i[0] for i in overall_cluster_schedule]
            #st.write(overall_sum_timings)
            #st.write(len(overall_sum_timings))
            #st.write(len(*overall_sum_timings))
            length_sums=[len(i) for i in overall_cluster_sum_timings]
            
            LCM2=math.lcm(*length_sums)

            #st.write(LCM2)
            overall_cluster_sum_timings_lcm=[i*int(LCM2/len(i)) for i in overall_cluster_sum_timings]
            
            #sum_timings=list(itertools.chain.from_iterable(overall_sum_timings))
            sum_timings=[sum(x) for x in zip(*overall_cluster_sum_timings_lcm)]
            #st.write(max(sum_timings))
            #st.write("# Overlap Type")
            #st.write(overlap_type)
            if overlap_type!="manual": 
                sum_timings=list(map(sum, zip(*sum_timings)))
            #st.write(sum_timings)
            #st.write((shift_factor))
            cluster_shift=min(overall_schedule)
            #schedule=[x-cluster_shift for x in schedule]
            #schedule=[x if x<=len(sum_timings) else x-len(sum_timings) for x in schedule]

            sum_timings=sum_timings[cluster_shift:]+sum_timings[:cluster_shift]
            time=np.arange(0, len(sum_timings))
            valve_numbers=sum_timings

            # Plot the orthogonal step function
            fig,ax=plt.subplots()

            for i in range(len(time) - 1):
                ax.plot([time[i], time[i + 1]], [valve_numbers[i], valve_numbers[i]], color='blue')
                ax.plot([time[i + 1], time[i + 1]], [valve_numbers[i], valve_numbers[i + 1]], color='blue')
                x = [time[i], time[i + 1], time[i + 1], time[i]]
                y = [valve_numbers[i], valve_numbers[i], 0, 0]
                ax.fill(x, y, color='lightblue', alpha=0.5)
                #ax.plot(sum_timings[:240], linestyle='-', color='black')
            #arrow_labels=data_table["Well No"].to_list()



            font_properties = FontProperties(family='serif', size=15, weight='normal', style='italic')
            for value in schedule:
                    ax.arrow(value,-0.2, 0, 0.2, head_width=2.5, head_length=0.1, fc='black', ec='black')
                
                
            y_ticks = range(0, int(max(sum_timings))+3,1)
                #x_ticks = range(0, min(len(sum_timings),241)+20,10)
            plt.yticks(y_ticks)
                #plt.xticks(x_ticks)
                
            ax.set_xlabel('Time in minute')
            ax.set_ylabel('Well overlap count')
            ax.grid(color='lightgray', linestyle='--')
            #plt.show()
                
            st.write("---")
            st.write("# All clusters well overlapping:")
            st.write("- Arrow marks cluster-wise start of cycle for one LCM cycle")
            st.pyplot(fig)
            plotly_fig = mpl_to_plotly(fig)
            if iv: 
                st.write("---")
                st.write("## Interactive View:")
                st.plotly_chart(plotly_fig)
            
            st.write("---")
            #st.write("---")
            st.write("# Time utilization for all clusters: ")
            fig1,ax1=plt.subplots(figsize=(10, 10))
            sum_timings_set=list(set(int(x) for x in sum_timings))
            sum_timings_frequency=[]
            for i in sum_timings_set:
                sum_timings_frequency.append(math.ceil(sum_timings.count(i)/len(sum_timings)*100*100)/100)
            
            mean=sum(sum_timings)/len(sum_timings)
            
            # Plot the data on the new axis
            ax1.plot(sum_timings_set, sum_timings_frequency)

            #annotations = [f'({xi}, {yi})' for xi, yi in zip(sum_timings_set, sum_timings_frequency)]
            #for annotation, xy in zip(annotations, zip(x, y)):
            #    ax1.annotate(annotation, xy)
            #    
            #    # Connect dotted orthogonal lines to the x and y axes
            #    ax.plot([xy[0], xy[1]], [xy[1], 0], linestyle='dotted', color='gray')
            #    ax.plot([xy[0], 0], [xy[0], xy[1]], linestyle='dotted', color='gray')

            # Plot the mean value as a dotted vertical line
            ax1.axvline(x=mean, linestyle='dotted', color='red', label='Mean is '+str(round(mean,2)))
            x_ticks=sum_timings_set
            #st.write(x_ticks)
            ax1.set_xticks(x_ticks)
            ax1.legend()
            ax1.grid(color='lightgray', linestyle='--')
            # Set labels for the axes and title
            ax1.set_xlabel('No. of gas injection wells')
            ax1.set_ylabel('Timing percentage of the day (%)')
            #ax1.set_title('Continuous Straight Line Plot')
            #print(data_table)
                # Show the plot
            left_col,centre_col,right_col=st.columns((0.2,1,0.3))
            with centre_col:
                st.pyplot(fig1)
        
            
                

            st.write("---")
            st.write('## Results:')
            overlap_value=int(max(sum_timings)) if max(sum_timings)>1 else "No Overlapping!"
            st.write("- Maximum well overlap: "+str(overlap_value))
            #st.write("- Maximum valve overlap count (in minutes) in a cycle: "+str(overlap_number))
            #if priority_number==0:
                #st.write("- For cluster wells overlapping with other wells has been avoided successfully")
            #else:
                #st.write("- Well overlapping is present for cluster wells, however, minimum overlapping scheduling is done.")
            st.write("---")
            ###########HERE 
    except ValueError: 
        #st.write(ValueError)
        st.warning("Data filled incorrectly! Please check again :)")


def contact():
    with st.container():
        st.write("---")
        st.header("Get In Touch With Us!")
        st.write("##")

        # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
        contact_form = """
        <form action="https://formsubmit.co/c0187a62200d807b7472b0e5695ba4e6" method="POST">
            <input type="hidden" name="_captcha" value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit">Send</button>
        </form>
        """
        left_column, right_column = st.columns(2)
        with left_column:
            st.markdown(contact_form, unsafe_allow_html=True)
        with right_column:
            st.empty()
        st.write("---")

def local_css():
    with open("style/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#Helper functions

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 105px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="white",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer():
    myargs = [
        "Made by ",
        link("https://instagram.com/amnn.sharma/", "@amansharma"),
        br(),
        link("https://buymeacoffee.com/amnnsharma", image('https://i.imgur.com/thJhzOO.png')),
    ]
    layout(*myargs)

footer()

def main():
    
    local_css()

    pages = {
        "Home": home,
        "Analysis": analysis,
        "Analysis (Cluster Wise)": cluster_wise_analysis,
        "Feedback":contact
    }

    st.sidebar.title("Navigation")
    page_selection = st.sidebar.radio("Go to", list(pages.keys()))

    if page_selection in pages:
        pages[page_selection]()

if __name__ == "__main__":
    
    main()
