//
// factorgraph-viz
//
// Visualizing factor graphs using d3-force.
//
// author: mbforbes
//
//
// factorgraph-viz
//
// Visualizing factor graphs using d3-force.
//
// author: mbforbes
//
/**
 * nodetype returns a function that will take FGNodes as arguments and return
 * whether they match the desired type.
 * @param desired
 */
function nodetype(desired) {
    return function (node) {
        return node.type === desired;
    };
}
/**
 * nodesubtype returns a function that will take FGNodes as arguments and return
 * whether they match the desired subtype.
 * @param desired
 */
function nodesubtype(desired) {
    return function (node) {
        // TODO: do we want to check the node's focus?
        // let focus = node.focus || false;
        let focus = false;
        return (!focus) && node.subtype === desired;
    };
}
/**
 * nodefocus returns whether a node is the node to focus on visually.
 * @param node
 */
function nodefocus(node) {
    return node.focus || false;
}
/**
 * textclass returns the class that should be applied to the text surrounding
 * the provided node.
 * @param node
 */
function textclass(node) {
    return node.type === 'rv' ? 'rvtext' : 'factext';
}
/**
 * nodename determines the text that is rendered next to a node.
 * @param node
 */
function nodename(node) {
    if (node.type == 'fac') {
        // maybe add extra info (e.g. sel pref fac is reversed)
        let specific = '';
        if (node.specific != null) {
            specific = ' [' + node.specific + ']';
        }
        return node.subtype + specific;
    }
    else {
        // rv
        return node.id;
    }
}
//
// factorgraph-viz
//
// Visualizing factor graphs using d3-force.
//
// author: mbforbes
//
//
// util.ts has a few helper functions, mostly regarding colorizing.
//
/// <reference path="node.ts" />
function argmax(arr) {
    if (arr.length < 1) {
        return -1;
    }
    let max_val = arr[0], max_idx = 0;
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}
function color(none, unsureColor, unsureCutoff, values, d) {
    if (d.weights == null) {
        return d3.color(none);
    }
    let max_idx = argmax(d.weights);
    let max_val = d.weights[max_idx];
    // clamp unsure ones to final value (hopefully something like grey)
    if (max_val < unsureCutoff) {
        return d3.color(unsureColor);
    }
    return d3.color(values[argmax(d.weights)]);
}
//
// factorgraph-viz
//
// Visualizing factor graphs using d3-force.
//
// author: mbforbes
//
//
// graph.ts defines the monster build(...) function for constructing the factor
// graph. It's full of closures as an excuse for accessing what are basically
// globals. I blame d3.
//
/// <reference path="config.ts" />
/// <reference path="node.ts" />
/// <reference path="util.ts" />
function appendText(svg) {
    let count = 1;
    return function (label, d) {
        if (d) {
            svg.append('g').append('text')
                .attr('transform', 'translate(20,' + count * 20 + ')')
                .text(label + ': ' + d);
            count += 1;
        }
    };
}
/**
 *
 * build is the central function of this codebase. It pareses the factor graph
 * data and constructs it.
 *
 * Note: the nodes here are technically FGNodes, but the horrendous type
 * massaging needed to make this work with d3's type hariness is not worth the
 * effort.
 * @param config
 * @param data
 */
function build(config, data) {
    let svg = d3.select("svg"), width = +svg.attr("width"), height = +svg.attr("height");
    // Debug logging. Can be nice as Chrome's console lets you interactively
    // explore the objects you're getting.
    console.log('Factor graph data:');
    console.log(data);
    function isolate(force, filter) {
        let initialize = force.initialize;
        force.initialize = function () { initialize.call(force, data.nodes.filter(filter)); };
        return force;
    }
    // TODO: We can actually extract most of this information. Stats should only
    // be used to provide additional info that can't be extracted from the graph
    // structure.
    let appeneder = appendText(svg);
    if (data.stats) {
        appeneder('random variables', data.stats.n_rvs);
        appeneder('factors', data.stats.n_facs);
        appeneder('focus', data.stats.focus);
        appeneder('correct', data.stats.correct);
    }
    let leftScale = config.position.leftScale;
    let rightScale = config.position.rightScale;
    let centerScale = config.position.centerScale;
    let sim = d3.forceSimulation(data.nodes)
        .force('charge', d3.forceManyBody().strength(-500))
        .force('link', d3.forceLink(data.links).id(function (d) { return d.id; }))
        .force('center', isolate(d3.forceCenter(width * centerScale, height / 2), nodefocus))
        .force('left', isolate(d3.forceX(width * leftScale).strength(config.position.leftStrength), nodesubtype(config.position.leftSubtype)))
        .force('right', isolate(d3.forceX(width * rightScale).strength(config.position.rightStrength), nodesubtype(config.position.rightSubtype)))
        .force('up', isolate(d3.forceY(config.position.upScale * height).strength(config.position.upStrength), nodesubtype(config.position.upSubtype)))
        .force('down', isolate(d3.forceY(config.position.downScale * height).strength(config.position.downStrength), nodesubtype(config.position.downSubtype)))
        .force('middle', d3.forceY(height / 2).strength(config.position.middleStrength))
        .on('tick', ticked);
    // use color config we've received to partially bind coloring function
    let colorize = color.bind(null, config.color.none, config.color.unsureColor, config.color.unsureCutoff, config.color.values);
    // new for svg --- create the objects directly; then ticked just modifies
    // their positions rather than drawing them.
    let link = svg.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(data.links)
        .enter().append("line")
        .attr("stroke", colorize);
    let text = svg.append('g')
        .selectAll('text')
        .data(data.nodes)
        .enter().append('text')
        .attr('class', textclass)
        .text(nodename);
    let node = svg.append("g")
        .attr("class", "nodes")
        .selectAll("circle")
        .data(data.nodes.filter(nodetype('rv')))
        .enter().append("circle")
        .attr("r", config.size.rv)
        .attr("fill", colorize)
        .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));
    let fac = svg.append("g")
        .attr("class", "facs")
        .selectAll("rect")
        .data(data.nodes.filter(nodetype('fac')))
        .enter().append("rect")
        .attr("fill", colorize)
        .attr("width", config.size.factor)
        .attr("height", config.size.factor)
        .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended));
    // Assumes RVs and factor are roughly the same size.
    let bigger = Math.max(config.size.rv, config.size.factor);
    function ticked() {
        link
            .attr("x1", function (d) { return d.source.x; })
            .attr("y1", function (d) { return d.source.y; })
            .attr("x2", function (d) { return d.target.x; })
            .attr("y2", function (d) { return d.target.y; });
        node
            .attr("cx", function (d) { return d.x; })
            .attr("cy", function (d) { return d.y; });
        fac
            .attr("x", function (d) { return d.x - config.size.factor / 2; })
            .attr("y", function (d) { return d.y - config.size.factor / 2; });
        text
            .attr("transform", function (d) {
            return "translate(" + (d.x + bigger) + "," + (d.y + 10) + ")";
        });
    }
    // The following functions allow for dragging interactivity. They're here
    // because they require access to variables defined in this function. (Well,
    // dragged() might not, but it fits with the others.)
    function dragsubject() {
        return sim.find(d3.event.x, d3.event.y);
    }
    function dragstarted() {
        if (!d3.event.active) {
            sim.alphaTarget(0.3).restart();
        }
        d3.event.subject.fx = d3.event.subject.x;
        d3.event.subject.fy = d3.event.subject.y;
    }
    function dragged() {
        d3.event.subject.fx = d3.event.x;
        d3.event.subject.fy = d3.event.y;
    }
    function dragended() {
        if (!d3.event.active) {
            sim.alphaTarget(0);
        }
        d3.event.subject.fx = null;
        d3.event.subject.fy = null;
    }
}
;
//
// factorgraph-viz
//
// Visualizing factor graphs using d3-force.
//
// author: mbforbes
//
//
// main.ts is where the execution begins.
//
/// <reference path="config.ts" />
/// <reference path="graph.ts" />
// Constants
let FG_NAME_ELEMENT_ID = 'fg-title';
let SVG_ELEMENT_ID = 'fg-svg';
let USER_INPUT_ID = 'userInput';
let SUGGESTIONS_ELEMENT_ID = 'suggestions';
let CONFIG_FILE = 'data/config/default.json';
// Globals (sorry).
let cacheConfig;
let cacheFactorgraphFns = [];
/**
 * Extracts general config and list of factorgraph file names. Calls preload.
 * @param config
 */
function prepreload(config) {
    cacheConfig = config;
    d3.json(config.data_filenames, preload);
}
/**
 * Saves the list of factor graph file names.
 * @param factorgraphFns
 */
function preload(factorgraphFns) {
    cacheFactorgraphFns = factorgraphFns;
}
/**
 * Helper to clear all children of a DOM node.
 * @param el
 */
function clearChildren(el) {
    while (el.firstChild) {
        el.removeChild(el.firstChild);
    }
}
/**
 * Removes everything from within the svg.
 */
function destroy() {
    clearChildren(document.getElementById(SVG_ELEMENT_ID));
}
/**
 * Loads factor graph found in `fn`.
 * @param fn
 */
function load(fn) {
    destroy();
    d3.json(fn, build.bind(null, cacheConfig));
}
/**
 * Loads factor graph found in `fn` if it's in our list of valid factor graph
 * names.
 * @param name
 */
function maybeLoad(name) {
    if (cacheFactorgraphFns.indexOf(name) != -1) {
        document.getElementById(FG_NAME_ELEMENT_ID).innerText = name;
        load(cacheConfig.data_dir + name + '.json');
    }
}
/**
 * Called every time the user text box changes its content.
 */
function userTypes() {
    let inp = document.getElementById(USER_INPUT_ID).value;
    // Prefix filter. Don't show anything with blank input
    let opts = [];
    if (inp.length > 0) {
        opts = cacheFactorgraphFns.filter(fn => fn.startsWith(inp));
    }
    // Clear any existing suggestions.
    let sug = document.getElementById(SUGGESTIONS_ELEMENT_ID);
    clearChildren(sug);
    // Add suggestions.
    for (let opt of opts) {
        let el = document.createElement('button');
        el.className = 'suggestion';
        el.innerText = opt;
        el.setAttribute('onclick', 'maybeLoad("' + opt + '");');
        sug.appendChild(el);
    }
}
/**
 * Called when the user submits the text box (presses enter or clicks button).
 * Always returns false so we don't do a post.
 */
function userSubmits() {
    maybeLoad(document.getElementById(USER_INPUT_ID).value);
    return false;
}
// execution starts here
d3.json(CONFIG_FILE, prepreload);
