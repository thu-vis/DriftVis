'use strict';
(function (my) {
  const that = {
    description: 'configure effect for scatterplot.',
  };
  my.EffectForScatterplot = that;
  that.densities = [];
  that.first_time = true;

  that.CircleVisualConfig = {
    big_size: 4,
    normal_size: 2,
  };

  that.drawNodes = function (visual_setting, node_list) {
    const container = visual_setting.container;
    // show_gmmlabel_color = visual_setting.show_gmmlabel_color,
    // animateAxisTime = visual_setting.animateAxisTime,
    // animateScatterMove = animateAxisTime;

    let animateTime;

    const _updateNodes = function (nodegroup) {
      // render points
      const nodeSlc = d3.select(this).selectAll('circle.scatter')
        .data(DataLoader.getVisibleData(node_list), d => d.t);
      if (that.first_time || DataLoader.actionFlags['direct-animation'] == true) {
        const time_config = AnimationControl.TimeConfig.quick.scatter;
        const _setNodeInfo = function (vn, _flag) {
          return (vn
            .attr('cx', (d) => (d.cx))
            .attr('cy', (d) => (d.cy))
            .attr('r', d => d.size)
            .attr('opacity', function (d) {
              return ((_flag && DataLoader.curDataDic[d.t] == 1) ? 1 : 0.5);
            }));
        };

        // quickTime = 1s
        _setNodeInfo(
          _setNodeInfo(nodeSlc.enter()
            .append('circle').classed('scatter', true).style('opacity', 0), false)
          .transition().delay(time_config.exit + time_config.update)
          .duration(time_config.enter)
          .style('opacity', 1)
          .style('stroke', 'gray')
          .style('stroke-width', 0.35),
          true);
        const afterNodeSlc = _setNodeInfo(nodeSlc.transition().delay(time_config.exit)
          .duration(time_config.update), true);
        nodeSlc.exit().transition().duration(time_config.exit)
          .style('opacity', 0).remove();
        afterNodeSlc.transition().duration(time_config.enter)
          .style('opacity', d => d.t < DataLoader.density_max ? 0 : 1);
        DataLoader.draw_batch_density(
          node_list,
          time_config.enter + time_config.update, time_config.exit
        );
        animateTime = time_config.exit + time_config.update + time_config.enter;
      } else {
        //    that.processNewData(nodeSlc);
        alert('Error');
      }
    };

    // following are designed for multiple scatterplots.
    const slc = container.selectAll('g.group').data([{
      type: 'node',
    }], (d) => (d.type));
    slc.enter().append('g').attr('class', 'group').each(_updateNodes);
    slc.each(_updateNodes);

    that.updateCursorGroup();

    if (that.first_time) {
      setTimeout(function () {
        ActionTrail.saveCurrentOverview(function () {
          ActionTrail.draw(AnimationControl.TimeConfig.action_trail.node_appear);
        });
      }, animateTime + 100);
    }
    that.first_time = false;
    return animateTime;
  };

  that.getAddDataAnimateInfo = function (enter_nodes, all_nodes) {
    const enterNodeDic = {};
    const cls2enterNodes = {};
    let allDic = {};
    enter_nodes.forEach((d) => {
      enterNodeDic[d.t] = d;
    });
    //nodeSlc.each((d) => {
    all_nodes.forEach((d) => {
      allDic[d.t] = d;
    });
    //allDic = Object.assign(allDic, enterNodeDic);
    // check correctness of label attribute
    const distrib = ModelLoader.distributions;
    const unmatchedNodes = [];
    for (const lkey of distrib.keys) {
      cls2enterNodes[lkey] = {
        idx: [],
        global_center: {
          x: 0,
          y: 0,
        },
      };
      distrib.components[lkey].idx.forEach((idx) => {
        cls2enterNodes[lkey].global_center.x += allDic[idx].cx;
        cls2enterNodes[lkey].global_center.y += allDic[idx].cy;

        if (enterNodeDic[idx] === undefined) {
          return;
        }
        cls2enterNodes[lkey].idx.push(idx);
        if (enterNodeDic[idx].label != lkey) {
          unmatchedNodes.push(idx);
        }
      });
      cls2enterNodes[lkey].global_center.x /= distrib.components[lkey].idx.length;
      cls2enterNodes[lkey].global_center.y /= distrib.components[lkey].idx.length;
    }
    console.log('unmatched nodes: ', unmatchedNodes);
    let event_idx = 1;
    // using bigTime as moveTime.
    const animateInfo = {};
    for (const lkey of distrib.keys) {
      const enterNodes = [];
      cls2enterNodes[lkey].idx.forEach((idx) => {
        const d = enterNodeDic[idx];
        enterNodes.push({
          t: d.t,
          x: d.cx,
          y: d.cy,
          trace: d.converted_trace,
        });
      });
      const groupResult = localGroupFunc(enterNodes);
      for (const key in groupResult) {
        let groupcenter = {
          x: 0,
          y: 0
        };
        let groupsize = groupResult[key].idx.length;
        let groupnodes = [];
        groupResult[key].idx.forEach((nodeID) => {
          const d = enterNodes[nodeID];
          groupnodes.push(d);
          groupcenter.x += d.x / groupsize;
          groupcenter.y += d.y / groupsize;
        });
        groupResult[key].idx.forEach((nodeID) => {
          const d = enterNodes[nodeID];
          animateInfo[d.t] = {
            index: event_idx,
            group: lkey,
            subgroup: 0,
            start: Object.assign({}, {
              x: (d.x + groupcenter.x) / 2 + (d.x - groupcenter.x) * (Math.random() - 0.5) * 0.2,
              y: (d.y + groupcenter.y) / 2 + (d.y - groupcenter.y) * (Math.random() - 0.5) * 0.2,
            }),
            end: Object.assign({}, {
              x: d.x,
              y: d.y,
            }),
          };
        });
        groupResult[key].idx.forEach((nodeID) => {
          const d = enterNodes[nodeID];
          const t = animateInfo[d.t];
          animateInfo[d.t].mid = {
            x: (t.start.x + t.end.x * 2) / 3,
            y: (t.start.y + t.end.y * 2) / 3
          };
        });
        if (groupnodes.length >= 20) {
          const subResult = kmeans(groupnodes, groupnodes.length / 10, 30);
          for (const key2 in subResult) {
            let subcenter = {
              x: 0,
              y: 0
            };
            let subsize = subResult[key2].idx.length;
            subResult[key2].idx.forEach((nodeID) => {
              const d = groupnodes[nodeID];
              subcenter.x += d.x / subsize;
              subcenter.y += d.y / subsize;
            });
            subResult[key2].idx.forEach((nodeID) => {
              const d = groupnodes[nodeID];
              const t = animateInfo[d.t];
              animateInfo[d.t].subgroup = key2;
              animateInfo[d.t].mid.x = (subcenter.x + t.end.x * 4) / 5;
              animateInfo[d.t].mid.y = (subcenter.y + t.end.y * 4) / 5;
            });
          }
        }
        event_idx += 1;
      }
    }
    all_nodes.forEach(function (d) {
      if (enterNodeDic[d.t] !== undefined) {
        return
      }
      animateInfo[d.t] = {
        start: {
          x: d.cx,
          y: d.cy,
        },
        end: {
          x: d.cx,
          y: d.cy,
        },
        index: 0, // original nodes should be added first.
      };
    });
    for (let t in animateInfo) {
      animateInfo[t].trace = allDic[t].converted_trace;
    }
    console.log('animateInfo', animateInfo)

    return animateInfo;
  };


  that.getRenderInSedimentation = function (coords, is_init) {
    is_init = (is_init === undefined) ? false : true;
    if (is_init) {
      return (function (nodeSlc, t1, t2) {
        nodeSlc.each(function (d) {
          let node = d3.select(this);
          node = node.transition().duration(t1).delay(t2)
            .attr('cx', (d) => (coords[d.t].x))
            .attr('cy', (d) => (coords[d.t].y))
            .style('fill', (d) => (d.light_color))
            .style('opacity', (d) => (d.t < DataLoader.density_max) ? 0 : 1)
          // if (d.t < DataLoader.density_max)
          //  node.style('opacity', 0);
        });
        //DataLoader.draw_batch_density(DataLoader.data.filter(d => 1), t1);//d.t <= DataLoader.historical_max));
/*      nodeSlc.exit().transition().duration(t1).delay(t2)
          .attr('cx', (d) => (coords[d.t].x))
          .attr('cy', (d) => (coords[d.t].y))
          .style('opacity', 0).remove();
        //nodeSlc.each(function (d) {
        //  let node = d3.select(this);
        //  node = node.transition().duration(t1).delay(t2)
        nodeSlc.transition().duration(t1).delay(t2)
          .attr('cx', (d) => (coords[d.t].x))
          .attr('cy', (d) => (coords[d.t].y))
          .style('fill', (d) => (d.light_color));
        //if (d.t <= DataLoader.historical_max) {
        //  node.style('opacity', 0).remove();
        //}
        //});
        */
        DataLoader.draw_batch_density(
          DataLoader.data, t1, 0);
        return nodeSlc.enter().append('circle').attr('class', 'scatter')
          .style('opacity', 0)
          .attr('point-id', d => d.t)
          .attr('cx', (d) => (coords[d.t].x))
          .attr('cy', (d) => (coords[d.t].y))
          .attr('r', d => d.size)
          .style('fill', (d) => (d.light_color))
          .style('stroke', 'gray')
          .style('stroke-width', 0.35)
          .on('mouseover', function (d) {
            return
            if (parseFloat(d3.select(this).style('opacity')) < 1e-8) {
              return;
            }
            that.handleCircleMouseOver(d, d3.select(this));
          })
          .on('mouseout', function (d) {
            return
            if (parseFloat(d3.select(this).style('opacity')) < 1e-8) {
              return;
            }
            that.handleCircleMouseOut(d, d3.select(this));
          });
      });
    } else {
      return (function (nodeSlc, duration) {
        nodeSlc.each(function (d) {
        //  d3.select(this).style('opacity', 1);
         // if (d.t <= DataLoader.density_max)
         //   d3.select(this).style('opacity', 0);
        });
        if (duration < 50) {
          return nodeSlc
            .attr('cx', (d) => (coords[d.t].x))
            .attr('cy', (d) => (coords[d.t].y));
        } else {
          return nodeSlc
            .transition().duration(duration)
            .attr('cx', (d) => (coords[d.t].x))
            .attr('cy', (d) => (coords[d.t].y));
        }
      });
    };
  }; {
    /*
  that.processNewData = function (nodeSlc) {
    // let bigTime = that.first_time ? 10 : 400;
    // let stepNodeCount = 50;
    // let appearTime = 5;

    const bigSize = that.CircleVisualConfig.big_size;
    //const normalSize = that.CircleVisualConfig.normal_size;
    const animateScatterMove = AnimationControl.TimeConfig.add_data.scatter.update;

    const enterNodeDic = {};
    const cls2enterNodes = {};
    let allDic = {};
    nodeSlc.enter().each((d) => {
      enterNodeDic[d.t] = d;
    });
    nodeSlc.each((d) => {
      allDic[d.t] = d;
    });
    allDic = Object.assign(allDic, enterNodeDic);
    // check correctness of label attribute
    const distrib = ModelLoader.distributions;
    const unmatchedNodes = [];
    for (const lkey of distrib.keys) {
      cls2enterNodes[lkey] = {
        idx: [],
        global_center: {
          x: 0,
          y: 0,
        },
      };
      distrib.components[lkey].idx.forEach((idx) => {
        cls2enterNodes[lkey].global_center.x += allDic[idx].cx;
        cls2enterNodes[lkey].global_center.y += allDic[idx].cy;

        if (enterNodeDic[idx] === undefined) {
          return;
        }
        cls2enterNodes[lkey].idx.push(idx);
        if (enterNodeDic[idx].label != lkey) {
          unmatchedNodes.push(idx);
        }
      });
      cls2enterNodes[lkey].global_center.x /= distrib.components[lkey].idx.length;
      cls2enterNodes[lkey].global_center.y /= distrib.components[lkey].idx.length;
    }
    console.log('unmatched nodes: ', unmatchedNodes);
    let event_idx = 1;
    // using bigTime as moveTime.
    const animateInfo = {};
    for (const lkey of distrib.keys) {
      const enterNodes = [];
      cls2enterNodes[lkey].idx.forEach((idx) => {
        const d = enterNodeDic[idx];
        enterNodes.push({
          t: d.t,
          x: d.cx,
          y: d.cy,
        });
      });
      const groupResult = localGroupFunc(enterNodes);
      for (const key in groupResult) {
        groupResult[key].idx.forEach((nodeID) => {
          const d = enterNodes[nodeID];
          const e = cls2enterNodes[lkey].global_center;
          animateInfo[d.t] = {
            index: event_idx,
            start: Object.assign({}, {
              x: (d.x + e.x) / 2 + (d.x - e.x) * (Math.random() - 0.5) * 0.2,
              y: (d.y + e.y) / 2 + (d.y - e.y) * (Math.random() - 0.5) * 0.2,
            }),
            end: Object.assign({}, {
              x: d.x,
              y: d.y,
            }),
          };
        });
        event_idx += 1;
      }
    }
    nodeSlc.each(function (d) {
      animateInfo[d.t] = {
        start: {
          x: d.cx,
          y: d.cy,
        },
        end: {
          x: d.cx,
          y: d.cy,
        },
        index: 0, // original nodes should be added first.
      };
    });

    const _setFinalNodeInfo = function (vn, _flag) {
      return (vn
        .attr('cx', (d) => (d.cx))
        .attr('cy', (d) => (d.cy))
        .attr('r', d => d.size)
        .attr('opacity', function (d) {
          return ((_flag && DataLoader.curDataDic[d.t] == 1) ? 1 : 0.5);
        }));
    };

    const frameTime = AnimationControl.TimeConfig.add_data.per_frame;
    const _step_cb = function (node_coords) {
      console.log('step_call_back');
      enterNodes.filter((d) => (node_coords[d.t] !== undefined))
        .style('opacity', 1)
        .transition().duration(frameTime)
        .attr('cx', (d) => (node_coords[d.t].x))
        .attr('cy', (d) => (node_coords[d.t].y));
    };
    const _final_cb = function () {
      // _setFinalNodeInfo(nodeSlc.enter().transition().duration(per_frame), true);
      // enterNodes.transition().duration(frameTime)
      enterNodes.transition()
        .duration(AnimationControl.TimeConfig.add_data.scatter.final_move)
        .attr('cx', (d) => (d.cx))
        .attr('cy', (d) => (d.cy))
        .attr('r', d => d.size);
    };

    // enter creation
    const enterNodes = nodeSlc.enter().append('circle').attr('class', 'scatter')
      .style('opacity', 0)
      .attr('cx', (d) => (animateInfo[d.t].start.x))
      .attr('cy', (d) => (animateInfo[d.t].start.y))
      .attr('r', d => d.size)
      .on('mouseover', function (d) {
        that.handleCircleMouseOver(d, d3.select(this));
      })
      .on('mouseout', function (d) {
        that.handleCircleMouseOut(d, d3.select(this));
      });
    // move original nodes
    _setFinalNodeInfo(nodeSlc.transition().duration(animateScatterMove), false);

    const _start = function () {
      AnimationControl.animateWithSedimentation(animateInfo,
        _step_cb,
        _final_cb,
      );
    };
    setTimeout(_start, animateScatterMove + 200);
  };
*/
  }

  // init cursor circle at the bottom.
  that.ensureCursorGroup = function () {
    let svg = d3.select('#scatterplot').select('svg');
    let cursor = svg.select('g.cursor-group');
    if (cursor.empty()) {
      cursor = svg.append('g').classed('cursor-group', true);
      SVGUtil.styleD3(cursor, {
        fill: 'transparent',
        //fill: 'red',
        'stroke-width': 0
      });
    }
    cursor.raise();
    return cursor;
  };

  that.updateCursorGroup = function () {
    let svg = d3.select('#scatterplot').select('svg');
    let node_list = [];
    svg.selectAll('circle.scatter').each((d) => {
      node_list.push(d)
    });
    //let g = that.ensureCursorGroup();
    //g.selectAll('circle.base').remove();
    //let nodeSlc = g.selectAll('circle.base').data(DataLoader.getVisibleData(node_list));
    let judgeR = 7;
    //nodeSlc.exit().remove();

    let s1 = svg.select('g.plot').attr('transform');
    if (s1.includes(',')) s1=s1.split(',');
    else s1 = s1.split(' ');
    let paddingX = parseFloat(s1[0].split('(')[1]);
    let paddingY = parseFloat(s1[1].split(')')[0]);

    //const _update = function (_slc) {
    //  _slc.attr('cx', (d) => (d.cx + paddingX))
    //    .attr('cy', (d) => (d.cy + paddingY))
    //    .attr('r', d => judgeR);
    //    //.on('mouseover', function (d) {
    //    // that.handleCircleMouseOver(d, d3.select(this));
    //    //})
    //    //.on('mouseout', function (d) {
    //    // that.handleCircleMouseOut(d, d3.select(this));
    //    //});
    //};

    //let nodeSlcEnter = nodeSlc.enter().append('circle').classed('base', true);
    //_update(nodeSlcEnter);
    ////_update(nodeSlc);

    //const node2base = {};
    //nodeSlc.each(function(d) {
    //  node2base[d.t] = d3.select(this);
    //});

    const _dis = (a, b) => ((a.cx - b.cx) * (a.cx - b.cx) + (a.cy - b.cy) * (a.cy - b.cy));

    //container.on('mousemove', function() {
    svg.on('mousemove', function () {
      if (DataLoader.current_view == 'density diff') {
        return;
      }
      let p = undefined,
        mouseCoord = {
          cx: d3.event.offsetX - paddingX,
          cy: d3.event.offsetY - paddingY
        }
      svg.selectAll('circle.scatter').each(function (d) {
        if (d.t <= DataLoader.density_max && 
          !BrushController.showed_idx.has(d.t)) { // ignore hide nodes.
          return;
        }
        let tempDis = _dis(d, mouseCoord);
        if (p === undefined || p.dis > tempDis) {
          p = Object.assign({
            vis: d3.select(this),
            dis: tempDis
          }, d);
        }
        d3.select(this).style('stroke-width', 0.35); // normal
        that.handleCircleMouseOut(d, d3.select(this));
      });
      if (p.dis < judgeR * judgeR) {
        p.vis.raise();
        that.handleCircleMouseOver(p.vis.datum(), p.vis);
        p.vis.style('stroke-width', 1);
      }
    });

    //// Correct coordinates
    //container.on('click', function() {
    //  container.append('circle').attr('cx', d3.event.offsetX-10)
    //    .attr('cy', d3.event.offsetY-10).attr('r', 5)
    //    .style('fill', 'red');
    //});
  };

  that.show_tfidf_words = function(word_dict) {
    let items = Object.keys(word_dict).map(function(key) {
        return [key, word_dict[key]];
      });
      items.sort((a,b) => (b[1]-a[1]));
      return (items.slice(0,10).map(x => x[0]).join(','));
  }

  that.renderAttrInTooltip = function (data, key) {
    const max = data.max[key],
      min = data.min[key],
      mean = data.mean[key],
      attr = data.value[key];
    //let text = key + ': ' + (attr - mean).toFixed(2) +
    //  '(' + (min - mean).toFixed(2) + '~' + (max - mean).toFixed(2) + ')' +
    //  ' + ' + mean.toFixed(2);
    //if (attr > max || attr < min) {
    //  return TooltipRender.text.bold(text);
    //} else {
    //  return TooltipRender.text.normal(text);
    //}
    let text = key + ': '+ mean.toFixed(2) + 
      '(' + (min).toFixed(2) + '~' + (max).toFixed(2) + ')';
    return TooltipRender.text.normal(text);  };

  let last_get_origin = 0
  that.handleCircleMouseOver = function (d, vn) {
    d._pop_show = true;
    if (d.is_waiting) return;
    d.is_waiting = true;
    const genTriggerPopover = function () {
      return function (data) {
        d.is_waiting = false;
        if (d._pop_over === undefined) {
          $(vn.node()).popover({
            container: 'body',
            trigger: 'manual',
            placement: 'top',
            html: true
          });
          d._pop_over = true;
        }
        if (d._pop_show) {
          let html = "";
          if ('current_batch_tfidf' in data) {
            //html += '<p>' + that.show_tfidf_words(data['current_batch_tfidf']) + '</p>';
            html += '<p>' + data['current_batch_tfidf'].slice(0, 10).map(x => x[0]).join(', ') + '</p>'
            for (let title of data.titles)
                html += '<p>' + title[0] + '</p>';
          } else {
          if (data.mean === undefined) {
            let keylist = data.value;
            if ('avgWind' in data.value) {
              keylist = {
                'temp': 0, 'minTemp': 0, 'maxTemp': 0, 'dewPoint': 0, 
                'avgWind': 0,'maxWind': 0,'seaLevelPressure': 0, 'visibility': 0 
              }
            } 
            for (let key in keylist) {
              if (typeof (data.value[key]) == "number") {
                html += '<p>' + key + ': ' + Number(data.value[key]).toFixed(2) + '</p>';
              } else {
                html += '<p>' + key + ': ' + data.value[key] + '</p>';
              }
            }
          } else {
            let keylist = data.value;
            if ('avgWind' in data.value) {
              keylist = {
                'temp': 0, 'minTemp': 0, 'maxTemp': 0, 'dewPoint': 0, 
                'avgWind': 0,'maxWind': 0,'seaLevelPressure': 0, 'visibility': 0 
              }
            } 
            for (let key in keylist) {
              if (data.mean[key] === undefined) {
                if (typeof (data.value[key]) == "number") {
                  html += '<p>' + key + ': ' + Number(data.value[key]).toFixed(2) + '</p>';
                } else {
                  html += '<p>' + key + ': ' + data.value[key] + '</p>';
                }
              } else {
                html += that.renderAttrInTooltip(data, key);
              }
            }
          }}
          vn.attr('data-content', html);
          $(vn.node()).popover('show');
        }
      };
    };
    DataLoader.get_origin(d.t, genTriggerPopover(vn));
  };

  that.handleCircleMouseOut = function (d, vn) {
    d._pop_show = false;
    if (d._pop_over) {
      $(vn.node()).popover('hide');
    }
  };

  const FU = {};
  FU.map = function (arr, func) {
    const len = arr.length;
    const res = new Array(len);
    for (let i = 0; i < len; ++i) {
      res[i] = func(arr[i], i);
    }
    return res;
  };

  const kmeans = function (node_list, num, round) {
    let cls = {};
    const per_num = node_list.length / num;
    node_list.forEach((d, i) => {
      const clsID = Math.floor(i / per_num);
      if (cls[clsID] === undefined) {
        cls[clsID] = {
          idx: [],
        };
      }
      cls[clsID].idx.push(i);
    });

    for (let round_i = 0; round_i < round; ++round_i) {
      for (const key in cls) {
        cls[key].center = {
          x: 0,
          y: 0,
        };
        cls[key].idx.forEach((nodeID) => {
          cls[key].center.x += node_list[nodeID].x;
          cls[key].center.y += node_list[nodeID].y;
        });
        cls[key].center.x /= cls[key].idx.length;
        cls[key].center.y /= cls[key].idx.length;
      }

      // new cluster
      const new_cls = {};
      node_list.forEach((d, i) => {
        let minCls = undefined;
        for (const key in cls) {
          if (minCls === undefined || ((l2dis2(d, cls[key].center) < l2dis2(d, cls[minCls].center)))) {
            minCls = key;
          }
        }
        if (new_cls[minCls] === undefined) {
          new_cls[minCls] = {
            idx: [],
          };
        }
        new_cls[minCls].idx.push(i);
      });
      cls = new_cls;
    }
    return cls;
  };

  const l2dis2 = function (node1, node2) {
    return (((node1.x - node2.x) ** 2) + ((node1.y - node2.y) ** 2));
  };

  const localGroupFunc = function (node_list) {
    return kmeans(node_list, Math.floor(node_list.length / 50), 30);
  };

  const TooltipRender = {};
  TooltipRender.text = {};
  TooltipRender.text.normal = function (text) {
    return '<p>' + text + '<p>';
  };
  TooltipRender.text.bold = function (text) {
    return '<p style="font-weight:bold;">' + text + '<p>';
  };
})(window);