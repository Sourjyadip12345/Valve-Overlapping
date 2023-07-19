from valve_overlapping_genetic_algo import *
import streamlit as st
import pandas as pd
from PIL import Image
from st_aggrid import AgGrid, GridUpdateMode, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


st.set_page_config(page_title="Valve Scheduling", page_icon=":alarm_clock:")

def home():
    img_glv = Image.open("images/glv.jpg")
    
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
            sum_timings=list(itertools.chain.from_iterable(sum_timings))
            #print("Maximum Overlap Value Count:",overlap_number)
            #print("Priority Value:",priority_number)
            #print("Valve schedule",schedule)
            #st.write(sum_timings)
            time=np.arange(0, len(sum_timings))
            valve_numbers=sum_timings

            # Plot the orthogonal step function
            fig,ax=plt.subplots(figsize=(12, 4))

            for i in range(len(time) - 1):
                ax.plot([time[i], time[i + 1]], [valve_numbers[i], valve_numbers[i]], color='blue')
                ax.plot([time[i + 1], time[i + 1]], [valve_numbers[i], valve_numbers[i + 1]], color='blue')
                x = [time[i], time[i + 1], time[i + 1], time[i]]
                y = [valve_numbers[i], valve_numbers[i], 0, 0]
                ax.fill(x, y, color='lightblue', alpha=0.5)
            #ax.plot(sum_timings[:240], linestyle='-', color='black')
            for value in schedule:
                ax.arrow(value,-0.2, 0, 0.2, head_width=0.5, head_length=0.3, fc='black', ec='black')

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
            ax.set_ylabel('Valve overlap count')
            ax.grid(color='lightgray', linestyle='--')
            #plt.show()
            
            st.write("---")
            st.write("## Indicative valve overlapping diagram:")
            st.pyplot(fig)
            #st.write(input_valves)

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
            data_table["Timings"]=timings
            #st.write(data_table.columns)
            data_table=data_table.set_index('Well No')
            #data_table.reset_index(drop=True, inplace=False)

            #print(data_table)
            
            st.write("---")
            st.write("## Scheduling table for valves:")
            st.write("- 9:00 AM is taken as reference time")
            with st.container():
                _,table=st.columns((0.1,2))
                with table:
                    st.write(data_table)

            st.write("---")
            st.write('## Results:')
            overlap_value=max(sum_timings) if max(sum_timings)>1 else "No Overlapping!"
            st.write("- Maximum valves overlap: "+str(overlap_value))
            #st.write("- Maximum valve overlap count (in minutes) in a cycle: "+str(overlap_number))
            if priority_number==0:
                st.write("- For priority wells overlapping with other wells has been avoided successfully :)")
            else:
                st.write("- Valve overlapping is inevitable with the given the given valve timings :( However, minimum overlapping scheduling is done.")

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
        "Contact":contact
    }

    st.sidebar.title("Navigation")
    page_selection = st.sidebar.radio("Go to", list(pages.keys()))

    if page_selection in pages:
        pages[page_selection]()

if __name__ == "__main__":
    
    main()
