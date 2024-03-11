'use strict';
(function (my) {
  const that = {
    description: 'configure animation for lineplot and scatterplot.',
  };
  my.AnimationControl = that;

  that.TimeConfig = {
    quick: {
      scatter: {
        exit: 300,
        update: 400,
        enter: 300,
      },
      line: {
        axis_change: 500,
        curve_change: 500,
      },
    },

    add_data: {
      per_frame: 30,
      density: {
        update: 2000,
      },
      scatter: {
        update: 1000,
        enter: undefined, // determined by number of nodes
        final_move: 300,
        appear: 100,
        total: 5000,
        move: 1200,
        adjust: 200,
      },
      line: {
        // axis_change: 500 // equal to scatter.update
        // curve_change: undefined // equal to time of scatter.enter
      },
    },

    action_trail: {
      node_appear: 300,
    }
  };

  that.animateAddData = function (after_func) {
    // update legend in scatterplot
    DataLoader.updateScatterCoords();
    DataLoader.refresh_scatter_color();

    let delayTime = AnimationControl.TimeConfig.add_data.density.update; 
    let totalTime = AnimationControl.TimeConfig.add_data.scatter.total;
    let frameTime = AnimationControl.TimeConfig.add_data.per_frame;

    /**
     * Animation before sedimentation.
     */
    // Scatterplot
    let animateNodeInfo = undefined;
    let startPosition = undefined;
    let endPosition = undefined;
    const visibleData = DataLoader.getVisibleData(DataLoader.curData);
    const scatterplotSlc = d3.select('#scatterplot').select('svg').select('g.plot')
      .selectAll('g.group');
    const enterScatterList = [];
    scatterplotSlc.each(function (d) {
      const scatterSlc = d3.select(this).selectAll('circle')
        .data(visibleData, (d) => (d.t));
      if (animateNodeInfo === undefined) { // here we assume data in all scatterplot should be same.
        animateNodeInfo = EffectForScatterplot.getAddDataAnimateInfo(
          scatterSlc.enter().data(), DataLoader.curData);
        startPosition = {};
        endPosition = {};
        DataLoader.curData.forEach((d) => {
          startPosition[d.t] = animateNodeInfo[d.t].start;
          endPosition[d.t] = animateNodeInfo[d.t].end;
        });
      }
      const render = EffectForScatterplot.getRenderInSedimentation(startPosition, true);
      enterScatterList.push(render(scatterSlc, delayTime, 0));
    });
    // Lineplot
    let curNodeCount = undefined;
    //console.log(BrushController.linecharts)
    for (const lc of BrushController.linecharts) {
      //EffectForLinechart.updateAxisWithNewData(lc, DataLoader.curData, delayTime);
      if (curNodeCount === undefined) {
        curNodeCount = lc.data.length;
      }
      ActionTrail.draw(delayTime);
      //EffectForLinechart.updatePathInSedimentation(lc, curNodeCount, delayTime);
    }

    let nodes = that.nodesWithSedimentation(animateNodeInfo);
    let maxRank = Math.max(...nodes.map(d => d.trace && d.trace.length || 0))

    nodes.forEach((d, i) => {
      d.delay = totalTime / nodes.length * i;
      if (i && d.group == nodes[i - 1].group) {
        d.delay2 = Math.max(100, nodes[i - 1].delay2 - (totalTime / nodes.length));
      } else {
        d.delay2 = 300;
      }
    })
    
    let coords = {}
    for (let d of nodes) {
      coords[d.id] = d
    }
    const scatter_appear = AnimationControl.TimeConfig.add_data.scatter.appear
    const scatter_move = AnimationControl.TimeConfig.add_data.scatter.move
    const scatter_adjust = AnimationControl.TimeConfig.add_data.scatter.adjust
    console.log('coords', coords)
    for (const enterScatter of enterScatterList) {
      const curScatter = enterScatter.filter((d) => {
        //console.log(d.t, coords[d.t]);
        return coords[d.t] !== undefined; });
      if (!nodes[0].trace) {
        curScatter
          .style('opacity', 0)
          .style('fill', d => d.color)
          .attr('r', 4)
          .attr('cx', d => coords[d.t].start.x)
          .attr('cy', d => coords[d.t].start.y)
          .transition().duration(scatter_appear).delay(d => coords[d.t].delay + delayTime + 100)
          .style('opacity', 1)
          //.transition().duration(scatter_move)
          .transition().duration(scatter_move / 2 * 3)
          .attr('cx', d => coords[d.t].mid.x)
          .attr('cy', d => coords[d.t].mid.y)
          .ease(d3.easeQuadOut)
          .transition().duration(scatter_move / 3).delay(d => coords[d.t].delay2)
          .attr('cx', d => coords[d.t].end.x)
          .attr('cy', d => coords[d.t].end.y)
          .ease(d3.easeLinear)
          .transition().duration(scatter_adjust)
          .delay(500)
          .style('fill', d => d.light_color)
          .attr('r', d => d.size)
      } else {
        const n_iter = nodes[0].trace.length
        curScatter.each(
          function(d){
            let el = d3.select(this)
              .style('opacity', 0)
              .style('fill', d.color)
              .attr('r', 4)
              .attr('cx', coords[d.t].trace[0].x)
              .attr('cy', coords[d.t].trace[0].y)
              .transition().duration(scatter_appear * 6)
              .delay(delayTime + 100 + scatter_move / n_iter * (n_iter - coords[d.t].trace.length))
              .style('opacity', 1)
          
            if (false)
            for (let i = 1; i < coords[d.t].trace.length; ++i) {
              el = el.transition().duration(scatter_move * 5 / n_iter)
                .attr('cx', coords[d.t].trace[i].x)
                .attr('cy', coords[d.t].trace[i].y)
            }
            el = el.transition().duration(scatter_move * 6)
              .attr('cx', coords[d.t].trace[coords[d.t].trace.length - 1].x)
              .attr('cy', coords[d.t].trace[coords[d.t].trace.length - 1].y)
    
            el = el.transition().duration(scatter_adjust * 3)
              .delay(500)
              .style('fill', d.light_color)
              .attr('r', d.size)
          }
        )
      }
    }

    let step = nodes.length * (frameTime / totalTime)
    let lastnum = 0
    let numStep = nodes.length / maxRank
    //console.log('numStep', numStep, nodes.length)
    function updateLineChart(num) {
      //console.log('add step', num, lastnum)
        if (Math.floor(num) > lastnum + numStep) {
          lastnum += numStep
          for (const lc of BrushController.linecharts) {
            EffectForLinechart.updateAxisWithNewData(lc,
              DataLoader.curData.slice(0, curNodeCount + Math.floor(num)), 
              500);
            setTimeout(() => EffectForLinechart.updatePathInSedimentation(lc, curNodeCount + Math.floor(num), undefined), 300)
        }
        if (num > nodes.length) {
          for (const lc of BrushController.linecharts) {
            setTimeout(() => {
              EffectForLinechart.updateAxisWithNewData(lc,DataLoader.curData, 50)
              EffectForLinechart.updatePathInSedimentation(lc, curNodeCount, 50)
            }, 500)
          }
          return;
        }
      }
      setTimeout(() => updateLineChart(num + step), frameTime);
    }
    setTimeout(() => updateLineChart(step), 100);
    setTimeout(after_func, totalTime + delayTime + 100);
  };

  /**
   * Args:
   * @node_info {node_t: {index, start:{x,y}, end:{x,y} } }
   */
  that.nodesWithSedimentation = function (node_info) {
    //console.log('node_info', node_info)
    const range = {
      x: undefined,
      y: undefined,
    };
    const index2node = {};
    const index2center = {};
    const group2center = {};
    for (const t in node_info) {
      const node = node_info[t];
      utils.ensureDic(index2node, node.index, []);
      utils.ensureDic(index2center, node.index, {
        x: 0,
        y: 0,
        group: node.group
      });
      index2center[node.index].x += node.end.x;
      index2center[node.index].y += node.end.y;
      index2node[node.index].push(t);
      if (range.x === undefined) {
        range.x = [0.9 * node.start.x, node.start.x * 1.1];
        range.y = [0.9 * node.start.y, node.start.y * 1.1];
      }
      range.x[0] = Math.min(range.x[0], Math.min(node.start.x, node.end.x));
      range.x[1] = Math.max(range.x[1], Math.max(node.start.x, node.end.x));
      range.y[0] = Math.min(range.y[0], Math.min(node.start.y, node.end.y));
      range.y[1] = Math.max(range.y[1], Math.max(node.start.y, node.end.y));
    }

    for (const k in index2center) {
      const g = index2center[k].group;
      const len = index2node[k].length;
      utils.ensureDic(group2center, g, {
        x: 0,
        y: 0,
        n: 0
      });
      group2center[g].x += index2center[k].x;
      group2center[g].y += index2center[k].y;
      group2center[g].n += len;
      index2center[k].x /= len;
      index2center[k].y /= len;
    }

    for (const k in group2center) {
      group2center[k].x /= group2center[k].n;
      group2center[k].y /= group2center[k].n;
    }

    const scale = {
      x: d3.scaleLinear().domain(range.x).range([0, 1]),
      y: d3.scaleLinear().domain(range.y).range([0, 1]),
    };

    for (const t in node_info) {
      const node = node_info[t];
      node.pstart = {};
      node.pend = {};
      for (const key of ['x', 'y']) {
        node.pstart[key] = scale[key](node.start[key]);
        node.pend[key] = scale[key](node.end[key]);
      }
    }

    const framePerIndex = 40;
    const frameRealTime = that.TimeConfig.add_data.per_frame;

    // index=1,2,3,... for new data
    let idxs = [];
    for (const key in index2node) {
      if (key == 0) {
        continue;
      }
      idxs.push(key);
    }
    idxs = idxs.sort((a, b) => {
      const group_a = index2center[a].group;
      const group_b = index2center[b].group;
      if (group_a != group_b) {
        return group_a > group_a ? 1 : -1
      } {
        const ax = index2center[a].x - group2center[group_a].x;
        const ay = index2center[a].y - group2center[group_a].y;
        const bx = index2center[b].x - group2center[group_b].x;
        const by = index2center[b].y - group2center[group_b].y;
        return ax * (by - ay) - (bx - ax) * ay
      }
    });

    let nodes = [];
    for (const k in index2node) {
      if (k != 0) {
        index2node[k] = index2node[k].sort((a, b) => {
          if (node_info[a].subgroup != node_info[b].subgroup) {
            return node_info[a].subgroup > node_info[b].subgroup ? 1 : -1
          }
          const ax = node_info[a].end.x - index2center[k].x;
          const ay = node_info[a].end.y - index2center[k].y;
          const bx = node_info[b].end.x - index2center[k].x;
          const by = node_info[b].end.y - index2center[k].y;
          return ax * (by - ay) - (bx - ax) * ay
        })
        nodes = nodes.concat(index2node[k].map(d => ({
          id: d,
          group: k * 128 + node_info[d].subgroup,
          start: node_info[d].start,
          trace: node_info[d].trace,
          mid: node_info[d].mid,
          end: node_info[d].end,
        })))
      }
    }


    return nodes;

    const status = {
      curIdxI: 0,
      curFrame: 0,
      moveFlag: {},
      appearList: [],
    };

    const _ticked = function () {
      if (status.curIdxI >= idxs.length) {
        let finalCount = 40;

        function _temp_ticked() {
          that.applyForceDuringSedimentation(pm, node_info, status.moveFlag);
          finalCount -= 1;
          if (finalCount > 0) {
            setTimeout(_temp_ticked, frameRealTime);
          } else {
            if (final_call_back !== undefined) {
              final_call_back();
            }
          }
        }

        setTimeout(_temp_ticked, frameRealTime);
        return;
      }

      const curIndex = idxs[status.curIdxI];

      // Uniformly assign nodes of current index into different frames.
      const curNodeNum = index2node[curIndex].length;
      const curNodeL = Math.min(curNodeNum, Math.floor((status.curFrame) / framePerIndex * curNodeNum));
      const curNodeR = Math.min(curNodeNum, Math.floor((status.curFrame + 1) / framePerIndex * curNodeNum));
      const coords = {}

      for (let i = curNodeL; i < curNodeR; ++i) {
        const node_t = index2node[curIndex][i];
        const node = node_info[node_t];
        coords[node_t] = {
          id: node_t,
          x0: node.start.x,
          y0: node.start.y,
          mid: node.mid,
          x1: node.end.x,
          y1: node.end.y,
        };
      }

      step_call_back({
        start: curNodeL,
        count: curNodeR - curNodeL,
        all: curNodeNum,
      }, coords); 

      status.curFrame += 1;
      if (status.curFrame == framePerIndex) {
        status.curIdxI += 1;
        status.curFrame = 0;
        status.appearList = []
      }
      setTimeout(_ticked, frameRealTime);
    };

    setTimeout(_ticked, frameRealTime);
  };

  that.applyForceDuringSedimentation = function (pm, node_info, moveFlag) {
    const xyDic = pm.getCircleCoords({
      x: function (z) {
        return z;
      },
      y: function (z) {
        return z;
      },
    });
    const endSpace = 0.05;

    for (const node_t in moveFlag) {
      // let m = pm.bodyDic[node_t].GetMass(),
      //    d1 = utils.l2dis2(xyDic[node_t], node_info[node_t].start),
      //    d2 = utils.l2dis2(xyDic[node_t], node_info[node_t].end),
      //    d3 = utils.l2dis2(node_info[node_t].start, node_info[node_t].end);
      // let r1 = d1 / (d3 + 1e-8),
      //    r2 = d2 / (d3 + 1e-8);
      // repulsive force = time / distance_ratio
      const tempBody = pm.bodyDic[node_t];
      if (moveFlag[node_t] < 0) {
        delete moveFlag[node_t];
        pm.world.DestroyBody(tempBody);
        // console.log("remove node", node_t);
        continue;
      }

      // // Vector from start to cur
      // let vecStart = new b2Vec2(xyDic[node_t].x - node_info[node_t].start.x,
      //    xyDic[node_t].y - node_info[node_t].start.y);
      // let dis1 = vecStart.Normalize();
      // Vector from cur to end
      const vecEnd = new b2Vec2(node_info[node_t].pend.x - xyDic[node_t].x,
        node_info[node_t].pend.y - xyDic[node_t].y);
      const dis2 = vecEnd.Normalize();

      // if (dis1 < localSpace) {
      //    vecStart.Multiply(30 * tempBody.GetMass());
      //    tempBody.ApplyForce(vecStart, tempBody.GetWorldCenter());
      // } else
      if (dis2 < endSpace) {
        // if (tempBody.GetType() != b2Body.b2_kinematicBody) {
        //    tempBody.SetType(b2Body.b2_kinematicBody);
        // }
        tempBody.SetLinearVelocity(new b2Vec2(0, 0));
        tempBody.SetPosition(new b2Vec2(node_info[node_t].pend.x,
          node_info[node_t].pend.y));
        // delete moveFlag[node_t];
        moveFlag[node_t] = -1;
        continue;
      } else { // if (dis2 < localSpace) {
        // if (tempBody.GetType() != b2Body.b2_kinematicBody) {
        //    tempBody.SetType(b2Body.b2_kinematicBody);
        // }

        const vecAbs = (new b2Vec2(node_info[node_t].pend.x - node_info[node_t].pstart.x,
          node_info[node_t].pend.y - node_info[node_t].pstart.y)).Normalize();
        const vec = new b2Vec2(vecEnd.x, vecEnd.y);
        vec.Multiply(vecAbs * 2); // 10000000);
        tempBody.SetLinearVelocity(vec);
        // } else if (dis1 < localSpace) {
        //    vecStart.Multiply(2000 * tempBody.GetMass());
        //    tempBody.ApplyForce(vecStart, tempBody.GetWorldCenter());
        // } else {
        //    //if (tempBody.GetType() != b2Body.b2_kinematicBody) {
        //    //    tempBody.SetType(b2Body.b2_kinematicBody);
        //    //}
        //    let vec = new b2Vec2(vecEnd.x, vecEnd.y);
        //    vec.Multiply(480);
        //    tempBody.SetLinearVelocity(vec);
      }
    }
  };

  const b2Vec2 = Box2D.Common.Math.b2Vec2;
  const b2BodyDef = Box2D.Dynamics.b2BodyDef;
  const b2Body = Box2D.Dynamics.b2Body;
  const b2FixtureDef = Box2D.Dynamics.b2FixtureDef;
  const b2Fixture = Box2D.Dynamics.b2Fixture;
  const b2World = Box2D.Dynamics.b2World;
  const b2MassData = Box2D.Collision.Shapes.b2MassData;
  // b2PolygonShape = Box2D.Collision.Shapes.b2PolygonShape,
  const b2CircleShape = Box2D.Collision.Shapes.b2CircleShape;

  that.getPhyModel = function () {
    const world = new b2World(
      new b2Vec2(0, 0) // new b2Vec2(0, 10) //gravity
      , true, // true //allow sleep
    );

    const info = {};
    info.world = world;
    info.bodyDic = {};

    const fixDef = new b2FixtureDef;
    fixDef.density = 1.0; // 密度
    fixDef.friction = 0; // 0.5; // 摩擦力
    fixDef.restitution = 0.2; // 弹性
    const bodyDef = new b2BodyDef;

    info.addNewCircle = function (id, x, y, r, is_static) {
      if (is_static) {
        bodyDef.type = b2Body.b2_staticBody;
      } else {
        // bodyDef.type = b2Body.b2_dynamicBody;
        bodyDef.type = b2Body.kinematicBody;
      }
      bodyDef.position.x = x;
      bodyDef.position.y = y;
      fixDef.shape = new b2CircleShape(r);
      const body = world.CreateBody(bodyDef);
      body.CreateFixture(fixDef);
      info.bodyDic[id] = body;
      return body;
    };

    info.step = function () {
      world.Step(
        1 / 60 // frame-rate
        , 10 // velocity iterations
        , 10, // position iterations
      );
      return info;
    };

    info.getCircleCoords = function (_scale1) {
      _scale1 = (_scale1 === undefined) ? info.scale1 : _scale1;
      const res = {};
      for (const key in info.bodyDic) {
        const xy = info.bodyDic[key].GetWorldCenter();
        res[key] = {
          x: _scale1.x(xy.x),
          y: _scale1.y(xy.y),
        };
      }
      return res;
    };

    return info;
  };

  const utils = {};
  utils.ensureDic = function (dic, key, value) {
    if (dic[key] === undefined) {
      dic[key] = value;
    }
  };

  utils.l2dis2 = function (node1, node2) {
    return Math.sqrt(((node1.x - node2.x) ** 2) + ((node1.y - node2.y) ** 2));
  };
})(window);