from shiny import App, ui, reactive, render, req
from punnett_square import PunnettSquare
from shinywidgets import output_widget, render_widget

app_ui = ui.page_sidebar(
    ui.sidebar(
        'Parental Genotypes',
        ui.input_text('parent_a', 'First parent', value='Aa'),
        ui.input_text('parent_b', 'Second parent', value='Aa'),
        ui.input_radio_buttons(
            'type',
            '',
            {'genotypes': 'Genotypes', 'phenotypes': 'Phenotypes'}
        ),
        ui.input_slider('plot_size', 'Plot size', value=400, min=100, max=1000, 
                        step=10),
        ui.input_slider('font_size', 'Font size', value=15, min=1, max=30)
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header('Punnett Square'),
            output_widget('plot'),
            full_screen=True,
        ),
        ui.card(
            ui.card_header('Frequency Table'),
            ui.output_data_frame('frequencies'),
            full_screen=True,
        ),
        col_widths=(8, 4)
    )
)

def server(input, output, session):
    @reactive.calc
    def square():
        req(len(input.parent_b()) == len(input.parent_a()))

        return PunnettSquare(input.parent_a(), input.parent_b())

    @render_widget
    def plot():
        return square().plotly_square(width=input.plot_size(),
                                      fontsize=input.font_size(),
                                      type=input.type())
    
    @render.data_frame
    def frequencies():
        table = square().freq_table(type=input.type())
        return render.DataGrid(table)

app = App(app_ui, server)