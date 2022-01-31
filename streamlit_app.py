from factorizetangle import *
from utils import *
import streamlit as st
import streamlit.components.v1 as components
import json

html = '''
<script src="https://d3js.org/d3.v5.js"></script>
<svg id="tangle" style="margin: 0 auto; display: block;"></svg>
<script>

function is_on_top(d){
    return d.indexOf("'") === -1;
}

function dot_to_int(d){
    let i = d.replace("'", "");
    if (is_on_top(d)) return Number(2*i - 2);

    return Number(2*i-1);
}

let hor_pad = 150;
let ver_pad = 80;

let inv = VALUE;
let n = inv.length;
let w = 1000 * n;
let h = 1000 * n;
const tangle = d3.select('#tangle')
                 .attr('width', '100%')
                 .attr('height', '100%')
                 .style('background-color', 'white');
                 //.attr("transform", "translate("+w/2+","+h/2+")");
let vertices = [];
for(i = 1; i <= n; i++){
    tangle.append("circle")
                    .attr("cx", hor_pad + 50*i)
                    .attr("cy", 30)
                    .style("fill","black")
                    .attr("r", 6);
    tangle.append("circle")
                    .attr("cx", hor_pad + 50*i)
                    .attr("cy", 30 + ver_pad)
                    .style("fill","black")
                    .attr("r", 6);
    
    tangle.append("text")
        .attr("x", hor_pad + 50*i - 5)
        .attr("y", 20)
        .text(i);
    
    tangle.append("text")
        .attr("x", hor_pad + 50*i - 5)
        .attr("y", 30 + ver_pad + 20)
        .text(i + "'");

    vertices.push([hor_pad + 50*i, 30]);
    vertices.push([hor_pad + 50*i, 30 + ver_pad]);
}

for(i = 0; i < n; i++){
    let a = inv[i][0];
    let b = inv[i][1];

    ai = dot_to_int(a);
    bi = dot_to_int(b);

    vax = vertices[ai][0];
    vay = vertices[ai][1];
    vbx = vertices[bi][0];
    vby = vertices[bi][1];

    let path = d3.path();
    path.moveTo(vax,vay);
    path.arc(vbx, vby,20,0,-3.14);
    
    if(is_on_top(a) && is_on_top(b)){
        let h_arc = (vax - vbx)/2;
        if (h_arc < -25){
            h_arc = (vax - vbx)/5;
        }

        tangle.append("path").attr('d', function(d){
            return ['M', vax, vay, 'A', (vax - vbx)/2, ',', h_arc, 0, 0, ',', 0, vbx, ',', vay].join(' ');
        }).style("fill", "none").attr("stroke", "black").style("stroke-width", 1);
    }else{
        if(!is_on_top(a) && !is_on_top(b)){

        let h_arc = (vax - vbx)/2;
        if (h_arc < -25){
            h_arc = (vax - vbx)/5;
        }
            tangle.append("path").attr('d', function(d){
                return ['M', vax, vay, 'A', (vax - vbx)/2, ',', h_arc, 0, 0, ',', 1, vbx, ',', vay].join(' ');
            }).style("fill", "none").attr("stroke", "black").style("stroke-width", 1);
        }else{
            tangle.append("line")
                .style("stroke-width", 1)
                .style("stroke", "black")
                .attr("x1", vax)
                .attr("y1", vay)
                .attr("x2", vbx)
                .attr("y2", vby); 
        }
    }


    
}

</script>'''



init()

st.write('''
# RNA to Tangle
''')

dotbracket = st.text_input("Dot-Bracket String", "...(..[..{...[...)...]...(...)...]..}...",key="dotbracket")
try:
    inv = dot_bracket_to_tangle(dotbracket)
except Exception as e:
    st.write(str(e))
else:

    n = len(inv)
    inv_str = inv_to_text(inv)

    if dotbracket != "":
        st.latex(inv_str + " \in \\mathcal{B}"+ "_{"+str(n)+"}")

        components.html(html.replace("VALUE", str(inv)))


        factors = factorize_reduce(inv)
        factors_tex = [f[0] + "_{" + f[1:] + "}" for f in factors]

        ltx_factors = " \circ ".join(factors_tex) 

        st.latex(ltx_factors)

        data = {
            "dot-bracket" : dotbracket,
            "tangle" : inv_str,
            "factor-list" : factors

        }

        json_obj = json.dumps(data, indent = 2) 

        st.download_button("Download", str(json_obj), file_name="tangle.json", mime="text/json")