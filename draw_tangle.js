function is_on_top(d){
    return d.indexOf("'") === -1;
}

function dot_to_int(d){
    let i = d.replace("'", "");
    if (is_on_top(d)) return Number(2*i - 2);

    return Number(2*i-1);
}

let hor_pad = 200;
let ver_pad = 100;

let inv = VALUE;
let n = inv.length;
let w = 1000 * n;
let h = 1000 * n;
const tangle = d3.select('#tangle')
                 .attr('width', w)
                 .attr('height', h)
                 .style('background-color', 'green');
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