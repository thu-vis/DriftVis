'use strict';
(function (my) {
  const _ = {
    description: 'Create line chart on svg.',
  };
  my.ActionTrail = _;

  _.save_svg_url = 'saveAsSVG';
  _.save_history_url = 'saveHistory';
  _.load_history_url = 'loadHistory';
  _.static_path = 'static/';
  _.nodeSettings = {
    h: 60,
    w: 100,
    padding_left: 45,
    //circleR: 10,
    //circleBigR: 13,
    model_node_rect: {
      width: 20,
      height: 20
    },
    model_node_step: 3
  };
  _.layoutSettings = {};

  _.init = function () {
    _.all = [];
    _.dataNodes = [];
    _.cur = _.newNode({
      type: 'start',
    });
    _.cur.nodeStatus = {};
    _.cur.show_size = DataLoader.show_size;
    _.cur.model_index = 0;
    DataLoader.data.forEach((d) => {
      _.cur.nodeStatus[d.t] = 0;
    });
    _.updateDataLoaderCurData();
    _.root = _.cur;
    _.updateTimeline();
    // _.draw();
  };

  _._ensureContainer = function () {
    const div = d3.select('#action-trail-plot');
    const margin = {
      top: 10,
      right: 10,
      bottom: 10,
      left: 10,
    };
    const width = div.node().getBoundingClientRect().width - margin.left - margin.right;
    const height = div.node().getBoundingClientRect().height - margin.top - margin.bottom;
    let svg = div.select('svg')
      .attr('width', width + margin.left)
      .attr('height', height + margin.top)
    let leftX = _.nodeSettings.padding_left;
    let root = svg.select('.shift-view')
    if (root.empty())
      root = svg
      .append('g')
      .attr('transform', 'translate(' + leftX + ',' + margin.top + ')')
      .attr('class', 'shift-view');
    //div.select('svg').append('text').attr("x", 10).attr("y", 20).text("Action Trail").attr('font-size', '18px');
    _.layoutSettings.scaleX = d3.scaleLinear()
      .domain([0, 1]).range([leftX, leftX + width]);
    // fix the height of action-trail-plot
    // div.style('height', (parseFloat(div.style('height')) + 4) + 'px'); // 2 is for border
    return root;
  };

  _.draw = function (animate_time) {
    return;

    animate_time = (animate_time === undefined) ? 0 : animate_time;
    const root = _._ensureContainer();
    let tempNodes = _.getDataForRenderingNode(_.timeline);
    _.layout(tempNodes, _.timeline);

    const tempDataNodes = tempNodes.filter(d => d.is_data_node);

    const _getModelNodeLayout = function (data_node, node, index) {
      let origin = {
        x: data_node.layout.x + _.nodeSettings.w / 2 + 10,
        y: data_node.layout.y - _.nodeSettings.h / 2
      };
      const rectSetting = _.nodeSettings.model_node_rect,
        rectStep = _.nodeSettings.model_node_step;
      let layout = {};
      layout.rect = Object.assign({
        x: origin.x + (rectSetting.width + 5) * Math.floor(index / rectStep + 1e-8),
        y: origin.y + (rectSetting.height + 5) * (index - Math.floor(index / rectStep + 1e-8) * rectStep)
      }, rectSetting);
      layout.text = {
        x: layout.rect.x + rectSetting.width / 2,
        y: layout.rect.y + rectSetting.height / 2
      };
      return layout;
    };

    let tempModelNodes = [];
    tempDataNodes.forEach((d) => {
      if (d.no < 0 || d.data.overview === undefined) {
        return;
      }
      tempModelNodes = tempModelNodes.concat(
        tempNodes.filter((dd) => (dd.data.model_index !== undefined && dd.data.dataRoot.no == d.no)).map((dd, i) => ({
          data_node: d,
          model_node: dd,
          model_index: i + 1,
          layout: _getModelNodeLayout(d, dd, i)
        })));
    });
    //console.log('model-node: ', tempModelNodes);

    let maxY = 0;
    let no2layout = {},
      topNodes = tempDataNodes;
    //topNodes = [];
    //tempNodes.forEach((d) => {
    tempDataNodes.forEach((d) => {
      no2layout[d.no] = d.layout;
      //if (d.is_data_node) {
      //  topNodes.push(d);
      //}
    });

    const _drawDataNode = function (x) {
      x.each(function (d) {
          const vn = d3.select(this);
          if (d.is_data_node) {
            _.drawSVGOverview(vn, d, d.data.overview, d.time_region.idxs.length);
          }
          // else {
          //  _.drawSVGOverview(vn, d, d.data.overview);
          //}
        })
        .on('dblclick', function (d) {
          _.notify('change', d.data);
          _.updateTimeline(); // kelei: Must update timeline before rendering action-trail.
          // TODO: if mouseout event not activate, then use this to highlight current data.
          //BrushController.reset_brush(DataLoader.curData, true);
        });
      return x;
    };

    const _drawNodeTransition = function (x) {
      return (x.attr('transform', function (d) {
        maxY = Math.max(maxY, (d.layout.y + _.nodeSettings.h / 2));
        return 'translate(' + (d.layout.x - _.nodeSettings.w / 2) + ',' + (d.layout.y - _.nodeSettings.h / 2) + ')';
      }));
    };

    {
      // const _drawEdge = function (x, is_new, _line_animate) {
      //   x.each(function (d) {
      //     let parentLayout = undefined;
      //     if (d.is_data_node) {
      //       if (preData[d.no] === undefined) {
      //         return;
      //       }
      //       parentLayout = preData[d.no].layout;
      //     } else {
      //       return;
      //       parentLayout = no2layout[d.data.parent.no];
      //     }
      //     const vn = d3.select(this);
      //     let line = undefined;
      //     if (is_new) {
      //       line = vn.append('line');
      //     } else {
      //       line = vn.select('line');
      //     }
      //     if (_line_animate !== undefined) {
      //       line = _line_animate(line);
      //     }
      //     if (d.is_data_node) {
      //       line = utils.attrD3(line, {
      //         x1: parentLayout.x + _.nodeSettings.w / 2,
      //         y1: parentLayout.y,
      //         x2: d.layout.x - _.nodeSettings.w / 2,
      //         y2: d.layout.y
      //       });
      //     } else {
      //       line = utils.attrD3(line, {
      //         x1: parentLayout.x,
      //         y1: parentLayout.y + _.nodeSettings.h / 2,
      //         x2: d.layout.x,
      //         y2: d.layout.y - _.nodeSettings.h / 2
      //       });
      //     }
      //     line = utils.styleD3(line, {
      //       'stroke-width': 1,
      //       'stroke': 'lightgrey',
      //     });
      //   });
      //   return x;
      // };

      //let preData = {};
      //for (let i = 1; i < topNodes.length; ++i) {
      //  preData[topNodes[i].no] = topNodes[i - 1];
      //}
    }

    const _animate_func = (x) => (x.transition().duration(animate_time));

    //const edgeSlc = root.selectAll('g.action-trail-edge').data(_.all, (x) => (x.no));
    //const edgeSlc = root.selectAll('g.action-trail-edge').data(tempNodes.filter(d => d.is_data_node), (x) => (x.no));
    //const edgeSlcEnter = _drawEdge(edgeSlc.enter().append('g').attr('class', 'action-trail-edge').style('opacity', 0), true);
    //edgeSlcEnter.transition().delay(animate_time).duration(10).style('opacity', 1);
    //_drawEdge(edgeSlc, false, _animate_func);
    //edgeSlc.exit().remove();

    //const nodeSlc = root.selectAll('g.action-trail-node').data(_.all, (x) => (x.no));
    const nodeDataSlc = root.selectAll('g.action-trail-node').data(tempDataNodes, (x) => (x.no));
    const nodeDataSlcEnter = _drawDataNode(nodeDataSlc.enter().append('g').attr('class', 'action-trail-node').style('opacity', 0));
    nodeDataSlc.exit().remove();
    _animate_func(_drawNodeTransition(nodeDataSlcEnter))
      .style('opacity', 1);
    _drawNodeTransition(_animate_func(_drawDataNode(nodeDataSlc)));

    const _drawModelNode = function (_slc, _animate) {
      let rect = _slc.select('rect');
      if (_animate !== undefined) {
        rect = _animate(rect);
      }
      rect = SVGUtil.attrD3(rect, {
        x: (d) => (d.layout.rect.x),
        y: (d) => (d.layout.rect.y),
        width: (d) => (d.layout.rect.width),
        height: (d) => (d.layout.rect.height)
      });
      rect = SVGUtil.styleD3(rect, {
        fill: (d) => (ModelLoader.model_colors[d.model_node.data.model_index]),
        stroke: (d) => ((_.cur.no == d.model_node.no) ? 'rgb(204,98,87)' : 'lightgrey'),
        'stroke-width': 1,
        r: 3
      });
      return _slc;
    };

    // render action-nodes with type == 'new-model'
    const nodeModelSlc = root.selectAll('g.action-model-node').data(tempModelNodes, (x) => (x.model_node.no));
    const nodeModelSlcEnter = nodeModelSlc.enter().append('g').classed('action-model-node', true).style('opacity', 0);
    nodeModelSlc.exit().remove();
    nodeModelSlcEnter.append('rect');
    nodeModelSlcEnter.append('text');
    nodeModelSlcEnter.on('mouseover', function (d) {
        d3.select(this).select('rect').style('stroke-width', 2);
        //_.highlightNodeInActionNode(d.model_node);
      })
      .on('mouseout', function (d) {
        d3.select(this).select('rect').style('stroke-width', 1);
        //BrushController.reset_brush();
      })
      //.on('click', function (d) {
      //  _.highlightNodeInActionNode(d.model_node);
      //})
      .on('dblclick', function (d) {
        _.notify('change', d.model_node.data);
        _.updateTimeline(); // kelei: Must update timeline before rendering action-trail.
      });
    _animate_func(_drawModelNode(nodeModelSlcEnter)).style('opacity', 1);
    _drawModelNode(nodeModelSlc, _animate_func);

    // add simple scroll
    const new_svg_height = maxY + 20;
    const svg = d3.select('#action-trail-plot').select('svg');
    if (new_svg_height > parseInt(svg.attr('height'))) {
      svg.attr('height', new_svg_height);
    }
  };

  _.newNode = function (info) {
    info.parent = _.cur;
    info.no = _.all.length;
    info.overview = undefined;
    info.backendInfo = {};
    // set data node
    if (info.type == 'start' || info.type == 'new-data') {
      info.parent = undefined; // action nodes for adding data are used as root of tree.
      info.dataIndex = _.dataNodes.length;
      _.dataNodes.push(info);
      info.dataRoot = info;

      // calculate nodeStatus & newNodes
      let lastNodeStatus = (_.dataNodes.length > 1 ? _.dataNodes[_.dataNodes.length - 2].nodeStatus : {});
      info.nodeStatus = {};
      info.newNodes = [];
      DataLoader.data.forEach((d) => {
        info.nodeStatus[d.t] = 0;
        if (lastNodeStatus[d.t] === undefined) {
          info.newNodes.push(d.t);
        }
      });
    } else {
      if (info.parent === undefined) {
        console.log("Error: non-data-node has not been set parent!");
      }
      info.dataRoot = info.parent.dataRoot;
    }

    // other process
    if (info.type == 'new-data') {
      // info.parent = _.all[0]; // new-data node connect to root.
    } else if (info.type == 'new-model') {
      info.nodeStatus = info.parent.nodeStatus;
      info.overview = info.parent.overview;
    }
    // else if (info.type == 'new-brush') {
    //     info.nodeStatus = {};
    //     for (let t in info.parent.nodeStatus) {
    //         if (info.parent.nodeStatus[t] !== undefined) {
    //             info.nodeStatus[t] = 0;
    //         }
    //     }
    //     DataLoader.selected_idx.forEach((item_t) => {
    //         info.nodeStatus[item_t] = 1;
    //     });
    // }
    // else if (info.type == 'select-feature-detail' || info.type == 'remove-feature-detail') {
    // }
    _.all.push(info);
    return info;
  };

  _.saveCurrentOverview = function (callback) {
    if (_.cur.overview !== undefined) {
      if (callback !== undefined) callback();
      return;
    }
    const req = new request_node(_.save_svg_url, (data) => {
      _.cur.overview = data.static_svg_path;
      if (callback !== undefined) callback();
    }, 'json', 'POST');
    req.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    req.set_data(_.getStatusOverview(_.cur));
    req.notify();
    console.log('save current overview completed.');
  };

  _.notify = function (event_type, data) {
    data = (arguments[1] === undefined) ? undefined : arguments[1];
    console.log('notify ', event_type, ' data=', data);

    _.clearOtherViewFocus();
    if (event_type == 'change') {
      const target = data;
      if (target === undefined) {
        console.log('Error: Please give the target!');
        alert('Error!');
      }
      _.cur = target;
      DataLoader.show_size = target.show_size;
      DataLoader.actionFlags['direct-animation'] = true;
      _.updateDataLoaderCurData();
      _.loadHistory(target.backendInfo['uuid'], function () {
        const time_config = AnimationControl.TimeConfig.quick.scatter;
        _.updateTimeline();
        _.draw(time_config.exit);
        _.updateOtherView();
      });
    } else
    if (event_type == 'new-data' || event_type == 'new-model') {
      _.cur = _.newNode({
        type: event_type,
      });
      _.cur.show_size = DataLoader.show_size;
      if (_.cur.type == 'new-model') {
        _.cur.model_index = data;
        // Since no need to prepare overview, update action-trail directly.
        _.draw(AnimationControl.TimeConfig.action_trail.node_appear);
      }
      // refresh views
      _.updateDataLoaderCurData();
      _.updateTimeline();
      _.updateOtherView();
      // Save status of backend and get 'uuid'.
      _.saveHistory();
    }
  };

  _.getStatusOverview = function (stat) {
    const svg = d3.select('#scatterplot').select('svg');

    const tempSVG = d3.select('body').append('svg').attr('id', 'tmpSVG').style('display', 'none').html(
      svg.html());
    tempSVG.selectAll('.brush').remove();
    const svg_html = tempSVG.html();
    tempSVG.html(null);
    return {
      svg: svg_html,
      area: {
        x: '0',
        y: '0',
        w: svg.attr('width'),
        h: svg.attr('height'),
      },
      no: 'action-overview-' + stat.no,
    };
  };

  _.clearOtherViewFocus = function () {
    for (const _info of BrushController.linecharts) {
      _info.line_brush_area.call(_info.line_brush.move, null);
    }
    // remove brush rect
    // for (const _info of BrushController.linecharts) {
    //   if (_info.line_brush_area !== undefined) {
    //     _info.line_brush_area.style('opacity', 0);
    //     SVGUtil.ensureSpecificSVGItem(
    //       _info.vn.root, 'rect', 'sync-select-rect').style('opacity', 0);
    //   }
    // }
    //        DataLoader.scatter_brush_area.style('opacity', 0);
  };

  _.updateOtherView = function () {
    if (DataLoader.actionFlags['direct-animation']) {
      DataLoader.draw_scatter_plot();
      DataLoader.draw_line_plot();
      DataLoader.actionFlags['direct-animation'] = false;
    } else if (_.cur.type == 'new-data') {
      AnimationControl.animateAddData(function () {
        // save scatter overview after animation
        EffectForScatterplot.updateCursorGroup();
        ActionTrail.saveCurrentOverview(ActionTrail.draw);
        // update the brush for scatterplot
        BrushController.update_scatter_brush();
        // update linechart grid
        DataLoader.updateGridLinechart();
      });
    } else {
      // nothing
    }
    ModelLoader.update_model_list();
  };

  let last_datasize = 0;
  _.updateDataLoaderCurData = function () {
    const new_data = [];
    const new_dic = {};
    const new_selected_idx = [];
    const flag = _.cur.nodeStatus;
    DataLoader.data.forEach((d) => {
      if (flag[d.t] !== undefined) {
        new_data.push(d);
        new_dic[d.t] = flag[d.t];
        if (flag[d.t] == 1) {
          new_selected_idx.push(d.t);
        }
      }
    });
    DataLoader.current_max = new_data[new_data.length - 1].t;
    if (DataLoader.current_max > last_datasize) {
      DataLoader.density_min = DataLoader.density_max;
      DataLoader.density_max = last_datasize;
      last_datasize = DataLoader.current_max;
    }
    DataLoader.historical_max = Math.max(0, DataLoader.current_max - DataLoader.show_size);
    DataLoader.curData = new_data;
    DataLoader.curDataDic = new_dic;
    BrushController.showed_idx = new Set(new_selected_idx);
    BrushController.selected_idx = new Set();
  };

  _.updateTimeline = function () {
    let droot = _.cur.dataRoot,
      curIndex = droot.dataIndex;
    let regions = [];

    let _new_region = function (_id, _x, _idxs, _expand) {
      regions.push({
        id: _id,
        x: _x,
        idxs: _idxs,
        expand: _expand
      });
    }

    if (curIndex > 1) {
      _new_region(-1, 0.2, d3.range(0, curIndex - 1), false);
    }
    if (curIndex > 0) {
      _new_region(-2, 0.5, [curIndex - 1], true);
    }
    _new_region(-3, 0.8, [curIndex], true);
    if (_.dataNodes.length > curIndex + 1) {
      _new_region(-4, 1, d3.range(curIndex + 1, _.dataNodes.length), false);
    }
    for (let i = 0; i < regions.length; ++i) {
      regions[i].w = regions[i].x - (i > 0 ? regions[i - 1].x : 0);
    }

    if (regions[regions.length - 1].x < 1) {
      let scale = d3.scaleLinear().domain([0, regions[regions.length - 1].x]).range([0, 1]);
      for (let i = 0; i < regions.length; ++i) {
        regions[i].x = scale(regions[i].x);
        regions[i].w = regions[i].x - (i > 0 ? regions[i - 1].x : 0);
      }
    }

    _.timeline = regions;
  };

  _.getDataForRenderingNode = function (regions) {
    let showFlag = {},
      region2node = {};
    for (let i = 0; i < regions.length; ++i) {
      if (regions[i].idxs.length == 1) {
        let data_node = _.dataNodes[regions[i].idxs[0]];
        showFlag[data_node.no] = regions[i];
      }
      region2node[regions[i].id] = [];
    }

    _.all.forEach((d) => {
      let data_node = d.dataRoot;
      if (showFlag[data_node.no] !== undefined) {
        if (showFlag[d.no] === undefined) {
          showFlag[d.no] = showFlag[data_node.no];
        }
        region2node[showFlag[d.no].id].push(d);
      }
    });

    let temp_nodes = [];
    regions.forEach((r) => {
      if (r.idxs.length == 1) {
        region2node[r.id].forEach((d) => {
          temp_nodes.push({
            no: d.no,
            data: d,
            time_region: r,
            is_data_node: (d.dataRoot === d)
          });
        });
      } else {
        temp_nodes.push({
          no: r.id,
          data: _.dataNodes[r.idxs[r.idxs.length - 1]],
          time_region: r,
          is_data_node: true
        });
      }
    });
    return temp_nodes;
  };

  _.getExpandTimeRegion = function () {
    let axisRange = [];
    _.timeline.forEach((r) => {
      if (r.expand) {
        axisRange.push({
          l: r.x - r.w,
          r: r.x
        });
      }
    });
    return axisRange;
  };

  /**
   * generate layout for nodes that showed.
   *
   * @param {*} render_nodes, two types:
   *    {no, data, time_region},
   *    {no, data:undefined, time_region} // stacked data node.
   * @param {*} regions, time regions.
   */
  _.layout = function (render_nodes, regions) {
    let region2node = {};
    render_nodes.forEach((node) => {
      if (region2node[node.time_region.id] === undefined) {
        region2node[node.time_region.id] = [];
      }
      region2node[node.time_region.id].push(node);
    });

    const nodes = [];
    const edges = [];
    const h = _.nodeSettings.h;
    const w = _.nodeSettings.w;
    // add virtual root to layout data nodes.
    let virtual_root = 'virtual_root';
    // add real action nodes
    render_nodes.forEach(function (node) {
      nodes.push({
        name: '' + node.no,
        h: h,
        w: w
      });
      if (node.is_data_node) {
        edges.push({
          start: virtual_root,
          end: '' + node.no
        });
      } else {
        edges.push({
          start: '' + node.data.parent.no,
          end: '' + node.no,
        });
      }
    });
    nodes.push({
      name: virtual_root,
      h: 1,
      w: 1
    });
    _._layoutDAG(nodes, edges);

    // shift xy as vertical center.
    const range = {
      min: undefined,
      max: undefined,
    };
    render_nodes.forEach(function (node, i) {
      node.layout = {
        x: nodes[i].x,
        y: nodes[i].y - 5,
      };
    });

    /*
     * Set action-node at the horizontal center of the corresponding time region in linechart.
     */
    const linechartInfo = BrushController.linecharts[0]; // Current there is only one linecharts.
    const xRange = linechartInfo.x_scale.range(),
      localScale = d3.scaleLinear().domain([0, 1]).range([xRange[0], xRange[xRange.length - 1]]);
    let regionShift = {};
    regions.forEach(function (r) {
      let center = localScale(r.x - r.w / 2),
        shift = undefined;
      region2node[r.id].forEach((d) => {
        if (d.is_data_node) {
          shift = center - d.layout.x;
        }
      });
      region2node[r.id].forEach((d) => {
        d.layout.x += shift;
      });
    });
  };

  // Description: calculate dag layout for nodes & edges.
  _._layoutDAG = function (nodes, edges) {
    // node format: name(str), w(float or int), h(float or int)
    // edge format: start(str: name of node), end(str: name of node)

    const that = this;
    const g = new dagre.graphlib.Graph();
    g.rankdir = 'TB';
    g.nodesep = 100;

    // Set an object for the graph label
    g.setGraph({});

    // Default to assigning a new object as a label for each new edge.
    g.setDefaultEdgeLabel(function () {
      return {};
    });

    nodes.forEach(function (node) {
      node.x = node.y = 0;
      g.setNode(node.name, {
        label: node.name,
        width: node.w,
        height: node.h * 0.67,
      }); // warning: reverse h and w
    });
    const edgeDic = {};
    edges.forEach(function (edge) {
      edgeDic[edge.start + '->' + edge.end] = edge;
      g.setEdge(edge.start, edge.end);
      // example: g.setEdge('cust_city','city_id', {label: '', lineInterpolate: 'monotone'});
    });
    dagre.layout(g);

    nodes.forEach(function (node) {
      const graphNode = g.node(node.name);
      node.x = graphNode.x;
      node.y = graphNode.y;
    });

    // g.edges().forEach(function (e) {
    //    var edge = edgeDic[e.v + "->" + e.w];
    //    edge.points = new Array();
    //    for (var i = 0; i < g.edge(e).points.length; i++) {
    //        var point = g.edge(e).points[i];
    //        var temp = point.x;
    //        point.x = (point.y + 10);
    //        point.y = temp;
    //        edge.points[i] = point;
    //    }
    // });
  };

  _.drawSVGOverview = function (vn, d, svg, boxNum = 1) {
    if (svg !== undefined) {
      // vn.selectAll('circle.action-circle').style('opacity', 0);
      let shiftPadding = 5;
      let rectSlc = vn.selectAll('rect.action-ow-bk').data(d3.range(boxNum - 1, -1, -1));
      rectSlc.exit();
      let _draw_box = function (rects) {
        //let box = utils.ensureSpecificSVGItem(vn, 'rect', 'action-ow-bk');
        rects.each(function (dd) {
          let box = d3.select(this);
          box = utils.attrD3(box, {
            x: -dd * shiftPadding,
            y: -dd * shiftPadding,
            width: _.nodeSettings.w,
            height: _.nodeSettings.h,
          });
          box = utils.styleD3(box, {
            'fill': 'white',
            'stroke-width': 2,
            'stroke': ((_.cur.dataRoot.no === d.data.no) ? 'rgb(204,98,87)' : 'lightgrey'),
          });
        });
      };
      _draw_box(rectSlc.enter().append('rect').attr('class', 'action-ow-bk'));
      _draw_box(rectSlc);

      //let image = utils.ensureSpecificSVGItem(vn,
      //  'image', 'action-ow-img');
      vn.select('image.action-ow-img').remove();
      let image = vn.append('image').classed('action-ow-img', true);
      image = utils.attrD3(image, {
        x: 1,
        y: 1,
        width: _.nodeSettings.w - 2,
        height: _.nodeSettings.h - 2,
      });
      image.attr('xlink:href', _.static_path + svg);

      // over & out
      vn.on('mouseover', function (d) {
        utils.styleD3(vn.selectAll('rect.action-ow-bk'), {
          'stroke-width': 2,
        });
        _.highlightNodeInActionNode(d);
      }).on('mouseout', function (d) {
        utils.styleD3(vn.selectAll('rect.action-ow-bk'), {
          'stroke-width': 1,
        });
        BrushController.reset_brush();
      }).on('click', function (d) {
        _.highlightNodeInActionNode(d);
      });
    }

    /**
     * @param: d, action-node.
     */
    _.highlightNodeInActionNode = function (d) {
      let new_nodes = [];
      if (d.no < 0) {
        d.time_region.idxs.forEach((rid) => {
          new_nodes = new_nodes.concat(_.dataNodes[rid].newNodes);
        });
      } else {
        new_nodes = d.data.newNodes;
      }
      BrushController.reset_brush(new_nodes);

    };
    // else {

    //    let circle = utils.ensureSpecificSVGItem(vn, 'circle', 'action-circle');
    //    circle = utils.attrD3(circle, {
    //        cx: 0,
    //        cy: 0,
    //        r: _.nodeSettings.circleR,
    //    });
    //    circle = utils.styleD3(circle, {
    //        fill: ((_.cur === d) ? 'red' : 'lightgrey')
    //    });
    //    // over & out
    //    vn.on("mouseover", function (d) {
    //        circle.attr("r", _.nodeSettings.circleBigR);
    //    }).on("mouseout", function (d) {
    //        circle.attr("r", _.nodeSettings.circleR);
    //    });
    // }
  };
  _.save = function (name) {
    const curNode = _.cur;
    const node = new request_node(_.save_history_url, (data) => {
      console.log('save history');
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'name': name,
    });
    node.notify();
  }
  _.saveHistory = function () {
    return;
    const curNode = _.cur;
    const node = new request_node(_.save_history_url, (data) => {
      console.log('save history');
      console.log(data['uuid']);
      // @kelei: bind uuid to actiontrail element
      // @bind event to save/load history
      curNode.backendInfo['uuid'] = data['uuid'];
    }, 'json', 'GET');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.notify();
  }

  _.loadHistory = function (uuid, callback) {
    const node = new request_node(_.load_history_url, (data) => {
      console.log('load history');
      ModelLoader.gmm_label = data['gmm_label'];
      ModelLoader.set_chunks(data['chunks']);
      ModelLoader.set_model(data['models']);
      DataLoader.fetch_label(function () {
        DataLoader.fetch_scatter_plot_data(callback);
      });
    }, 'json', 'POST');
    node.set_header({
      'Content-Type': 'application/json;charset=UTF-8',
    });
    node.set_data({
      'uuid': uuid,
    });
    node.notify();
  }

  const utils = {};
  utils.attrD3 = function (vnode, info) {
    for (const x in info) {
      vnode = vnode.attr(x, info[x]);
    }
    return vnode;
  };
  utils.styleD3 = function (vnode, info) {
    for (const x in info) {
      vnode = vnode.style(x, info[x]);
    }
    return vnode;
  };

  utils.visualNode2svgHtml = function (vn) {
    const w = vn.attr('width');
    const h = vn.attr('height');
    htmlText = '<svg width="' + w + '" height="' + h + '" viewBox="0 0 ' + h + ' ' + w + '">' + vn.html() + '</svg>';
    return (htmlText.replace(/"/g, '\''));
  };

  utils.svgHtmlChangeSize = function (svg_html, w, h) {
    const d = d3.select('body').append('div').style('display', 'none');
    d.html(svg_html);
    d.select('svg').attr('width', w).attr('height', h);
    const resultHtml = d.html().replace(/"/g, '\'');
    d.remove();
    return resultHtml;
  };

  utils.ensureSpecificSVGItem = function (_vn, _type, _cls) {
    let box = _vn.select(_type + '.' + _cls);
    if (box.empty()) {
      box = _vn.append(_type).attr('class', _cls);
    }
    return box;
  };
})(window);