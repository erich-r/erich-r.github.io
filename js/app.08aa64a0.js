(function(e){function t(t){for(var s,a,r=t[0],c=t[1],l=t[2],d=0,p=[];d<r.length;d++)a=r[d],Object.prototype.hasOwnProperty.call(o,a)&&o[a]&&p.push(o[a][0]),o[a]=0;for(s in c)Object.prototype.hasOwnProperty.call(c,s)&&(e[s]=c[s]);h&&h(t);while(p.length)p.shift()();return n.push.apply(n,l||[]),i()}function i(){for(var e,t=0;t<n.length;t++){for(var i=n[t],s=!0,r=1;r<i.length;r++){var c=i[r];0!==o[c]&&(s=!1)}s&&(n.splice(t--,1),e=a(a.s=i[0]))}return e}var s={},o={app:0},n=[];function a(t){if(s[t])return s[t].exports;var i=s[t]={i:t,l:!1,exports:{}};return e[t].call(i.exports,i,i.exports,a),i.l=!0,i.exports}a.m=e,a.c=s,a.d=function(e,t,i){a.o(e,t)||Object.defineProperty(e,t,{enumerable:!0,get:i})},a.r=function(e){"undefined"!==typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},a.t=function(e,t){if(1&t&&(e=a(e)),8&t)return e;if(4&t&&"object"===typeof e&&e&&e.__esModule)return e;var i=Object.create(null);if(a.r(i),Object.defineProperty(i,"default",{enumerable:!0,value:e}),2&t&&"string"!=typeof e)for(var s in e)a.d(i,s,function(t){return e[t]}.bind(null,s));return i},a.n=function(e){var t=e&&e.__esModule?function(){return e["default"]}:function(){return e};return a.d(t,"a",t),t},a.o=function(e,t){return Object.prototype.hasOwnProperty.call(e,t)},a.p="/";var r=window["webpackJsonp"]=window["webpackJsonp"]||[],c=r.push.bind(r);r.push=t,r=r.slice();for(var l=0;l<r.length;l++)t(r[l]);var h=c;n.push([0,"chunk-vendors"]),i()})({0:function(e,t,i){e.exports=i("56d7")},1960:function(e,t,i){"use strict";var s=i("5fb3"),o=i.n(s);o.a},"1be8":function(e,t,i){},"489f":function(e,t,i){e.exports=i.p+"media/mlgTick.60cb5a68.ogg"},"56d7":function(e,t,i){"use strict";i.r(t);i("e260"),i("e6cf"),i("cca6"),i("a79d");var s=i("2b0e"),o=function(){var e=this,t=e.$createElement,i=e._self._c||t;return i("v-app",[i("v-navigation-drawer",{attrs:{app:"",temporary:""},model:{value:e.drawer,callback:function(t){e.drawer=t},expression:"drawer"}},[i("v-list-item",[i("v-list-item-content",[i("v-list-item-title",{staticClass:"title"},[e._v(" "+e._s(e.isLoggedIn?e.account.username:"Anonymous")+" "),i("v-btn",{attrs:{icon:""},on:{click:function(t){e.drawer=!e.drawer}}},[i("v-icon",[e._v("mdi-chevron-left")])],1)],1),i("v-list-item-subtitle",[e._v("Menu")])],1)],1),i("v-divider"),i("v-list",[e._l(e.drawerItems,(function(t){return i("v-list-item",{key:t.title,attrs:{to:t.link,link:""}},[i("v-list-item-icon",[i("v-icon",[e._v(e._s(t.icon))])],1),i("v-list-item-content",[i("v-list-item-title",[e._v(e._s(t.title))])],1)],1)})),i("v-list-item",{attrs:{link:""},on:{click:function(t){return e.auth()}}},[i("v-list-item-icon",[i("v-icon",[e._v(e._s(e.isLoggedIn?"exit_to_app":"account_circle"))])],1),i("v-list-item-content",[i("v-list-item-title",[e._v(e._s(e.isLoggedIn?"Logout":"Login"))])],1)],1)],2)],1),i("v-app-bar",{attrs:{dense:"",app:"",color:"primary"}},[i("v-app-bar-nav-icon",{on:{click:function(t){t.stopPropagation(),e.drawer=!e.drawer}}}),i("v-toolbar-title",[e._v("PPII")])],1),i("v-content",{staticClass:"content"},[i("router-view",{ref:"view"}),i("v-snackbar",{attrs:{timeout:4e3,color:"info"},model:{value:e.events.sfida.arrivata,callback:function(t){e.$set(e.events.sfida,"arrivata",t)},expression:"events.sfida.arrivata"}},[e._v("Sei stato sfidato!")]),i("v-snackbar",{attrs:{timeout:4e3,color:"info"},on:{click:function(t){return e.goToEvent(!1)}},model:{value:e.events.partitaIniziata.arrivata,callback:function(t){e.$set(e.events.partitaIniziata,"arrivata",t)},expression:"events.partitaIniziata.arrivata"}},[e._v("Hai inziato una partita!")])],1)],1)},n=[],a=(i("b0c0"),i("d3b7"),i("ac1f"),i("3ca3"),i("841c"),i("ddb0"),i("2b3d"),i("96cf"),i("1da1")),r={name:"App",data:function(){return{drawer:!1,drawerItems:[{title:"Homepage",icon:"home",link:"/"},{title:"Practice",icon:"school",link:"/game"},{title:"Play",icon:"games",link:"/find_game"}],isLoggedIn:!1,account:{id:"",username:"",online:!0,perfs:{blitz:{games:0,rating:0,rd:0,prog:0},puzzle:{games:0,rating:0,rd:0,prog:0},atomic:{games:0,rating:0,rd:0,prog:0,prov:!1},ultraBullet:{games:0,rating:0,rd:0,prog:0,prov:!1},bullet:{games:0,rating:0,rd:0,prog:0},correspondence:{games:0,rating:0,rd:0,prog:0,prov:!1},classical:{games:0,rating:0,rd:0,prog:0,prov:!1},rapid:{games:0,rating:0,rd:0,prog:0,prov:!1}},createdAt:0,profile:{country:"",location:"",bio:"",firstName:"",lastName:"",links:""},seenAt:0,playTime:{total:0,tv:0},language:"",url:"",nbFollowing:0,nbFollowers:0,completionRate:0,count:{all:0,rated:0,ai:0,draw:0,drawH:0,loss:0,lossH:0,win:0,winH:0,bookmark:0,playing:0,import:0,me:0},followable:!1,following:!1,blocking:!1,followsYou:!1},events:{sfida:{arrivata:!1,value:""},partitaIniziata:{arrivata:!1,value:""}}}},methods:{auth:function(){this.isLoggedIn},goToEvent:function(e){e||this.$router.push({name:"game",params:{id:this.events.partitaIniziata.value}})},getAccountData:function(){var e=Object(a["a"])(regeneratorRuntime.mark((function e(){var t,s;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return t=i("bc3a"),e.next=3,t.get("/api/account",{baseURL:"https://lichess.org/",headers:{Authorization:"Bearer "+this.$store.getters.tokens}});case 3:s=e.sent,this.account=s.data;case 5:case"end":return e.stop()}}),e,this)})));function t(){return e.apply(this,arguments)}return t}(),listenForEvents:function(){var e=Object(a["a"])(regeneratorRuntime.mark((function e(){var t,s=this;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:t=i("048a"),fetch("https://lichess.org/api/stream/event",{headers:{Authorization:"Bearer "+this.$store.getters.tokens}}).then((function(e){return t(e.body)})).then((function(e){var t,i=e.getReader();i.read().then(t=function(e){e.done||(s.gotEvent(e.value),i.read().then(t))})})).catch((function(e){window.console.log("Errore: "+e)}));case 2:case"end":return e.stop()}}),e,this)})));function t(){return e.apply(this,arguments)}return t}(),gotEvent:function(e){var t=e.type;"gameStart"==t&&("game"==this.$route.name?this.$router.push({name:"game",params:{id:e.game.id}}):(this.events.partitaIniziata.arrivata=!0,this.events.partitaIniziata.value=e.game.id))}},created:function(){var e=new URLSearchParams(window.location.search),t=e.get("signout");if(null!=t)return this.$store.dispatch("LOGOUT","token"),history.pushState({},null,"/"),void(this.isLoggedIn=!1);var i=e.get("tok"),s=null==this.$store.getters.tokens;s?null!=i&&(this.$store.dispatch("LOGIN",i),history.pushState({},null,"/")):null!=i&&history.pushState({},null,"/#"),null==this.$store.getters.tokens?this.isLoggedIn=!1:(this.getAccountData(),this.listenForEvents(),this.isLoggedIn=!0)}},c=r,l=(i("5c0b"),i("2877")),h=i("6544"),d=i.n(h),p=i("7496"),u=i("40dc"),f=i("5bc1"),v=i("8336"),m=i("a75b"),g=i("ce7e"),_=i("132d"),b=i("8860"),y=i("da13"),w=i("5d23"),k=i("34c3"),x=i("f774"),I=i("2db4"),P=i("2a7f"),$=Object(l["a"])(c,o,n,!1,null,null,null),L=$.exports;d()($,{VApp:p["a"],VAppBar:u["a"],VAppBarNavIcon:f["a"],VBtn:v["a"],VContent:m["a"],VDivider:g["a"],VIcon:_["a"],VList:b["a"],VListItem:y["a"],VListItemContent:w["a"],VListItemIcon:k["a"],VListItemSubtitle:w["b"],VListItemTitle:w["c"],VNavigationDrawer:x["a"],VSnackbar:I["a"],VToolbarTitle:P["a"]});var B=i("2f62"),S=i("0e44"),C=i("8c4f"),T=i("2b27"),O=i.n(T),E=function(){var e=this,t=e.$createElement,i=e._self._c||t;return i("v-row",{staticClass:"cnt",staticStyle:{"padding-bottom":"2%"}},[i("v-col",{style:e.is_mobile?"":"padding-right:0px",attrs:{cols:"12",md:"7"}},[i("div",{style:"padding: "+e.board_padding+";"},[i("div",{staticClass:"chessboard"},[i("div",{ref:"chessBoard",staticClass:"wrap"},[i("div",[i("div",{staticClass:"helper"},[i("div",{staticClass:"ctr"},[i("div",{staticClass:"cbrd",on:{mousedown:e.got_selected,mousemove:e.while_selected,mouseup:e.stopDrag,touchstart:e.got_tapped}},[e._l(e.list_pieces,(function(t){return i("Piece",{key:t.id,ref:t.id,refInFor:!0,class:t.type+t.color+" "+t.id,attrs:{position:e.setPieceLocationInBoard(t.position.x,t.position.y),pieceName:t.type+t.color,pieceId:t.id}})})),e._l(e.valid_moves,(function(t,s){return i("Square",{key:s+t.to,attrs:{position:e.setPieceLocationInBoard(e.fileToNumber(t.to[0]),t.to[1]),attacked:null!=e.validator.get(t.to)}})}))],2)])])])])]),i("v-btn",{staticStyle:{"margin-top":"2%"},attrs:{tile:!0,color:"warning"},on:{click:function(t){return e.resetChessboard()}}},[e._v("RESET")])],1)]),i("v-col",{style:e.is_mobile?"padding:0 10%":"padding-left:0%",attrs:{cols:"12",md:"5"}},[i("h3",{staticStyle:{"text-align":"center"}},[e._v("Mosse")]),i("div",{class:e.is_mobile?"movesbox_mob":"movesbox"},[e._l(e.moves_history,(function(t,s){return["w"==t.color?i("span",{key:"n"+s,staticClass:"n_move",class:e.is_mobile?"move_mob":"move_number"},[e._v(e._s(e.moves_history.indexOf(t)/2+1))]):e._e(),i("span",{key:"m"+s,class:e.is_mobile?"move_mob":"move"},[e._v(e._s(t.san))])]}))],2)])],1)},V=[],A=(i("4160"),i("caad"),i("c975"),i("a434"),i("5377"),i("2532"),i("1276"),i("159b"),function(){var e=this,t=e.$createElement,i=e._self._c||t;return i("div",{class:"square "+e.getClass(),style:{transform:"translate("+this.piece_height+"%,"+this.piece_width+"%)"}})}),M=[],j={name:"Square",props:{position:{x:String,y:String},attacked:Boolean},data:function(){return{piece_height:this.position.x,piece_width:this.position.y,is_attack:this.attacked}},methods:{getClass:function(){return this.is_attack?"attacked":"not_attacked"}}},R=j,Y=(i("a532"),Object(l["a"])(R,A,M,!1,null,null,null)),X=Y.exports,N=function(){var e=this,t=e.$createElement,i=e._self._c||t;return i("div",{style:[this.is_selected?{top:this.mouse_top_pos+"px",left:this.mouse_left_pos+"px",zIndex:3}:{transform:"translate("+this.piece_height+"%,"+this.piece_width+"%)"}]})},D=[],F=(i("a9e3"),{name:"Piece",props:{position:{x:String,y:String},pieceName:String,pieceId:Number},data:function(){return{id:this.pieceId,name:this.pieceName,is_selected:!1,mouse_top_pos:0,mouse_left_pos:0,piece_height:this.position.x,piece_width:this.position.y}},watch:{position:function(e){this.piece_height=e.x,this.piece_width=e.y}},methods:{got_selected:function(){this.is_selected=!0,this.$el.classList.toggle("selected")},got_unselected:function(){this.$el.classList.toggle("selected"),this.is_selected=!1},set_mouse_pos:function(e,t){this.mouse_top_pos=e,this.mouse_left_pos=t},setPosition:function(e,t){this.piece_height=t,this.piece_width=e}}}),z=F,q=(i("d9ae"),Object(l["a"])(z,N,D,!1,null,"62053cc8",null)),G=q.exports,H=i("c5ad"),U=null,K={name:"Chessboard",components:{Piece:G,Square:X},data:function(){return{turn:!0,is_in_check:!1,has_piece_selected:!1,piece_selected:0,list_pieces:[],valid_moves:[],moves_history:[],validator:new H,board_side:!0,onPiecePlaceAudio:!1,debug:"",isTouchEvent:!1}},computed:{is_mobile:function(){return this.$vuetify.breakpoint.smOnly||this.$vuetify.breakpoint.xsOnly},board_padding:function(){var e="";switch(this.$vuetify.breakpoint.name){case"xs":e="2%";break;case"sm":e="0px 20% 0px 20%";break;case"md":e="0px 0px 0% 20%";break;case"lg":e="0px 0px 0% 20%";break;case"xl":e="0px 0px 0% 10%";break}return e}},mounted:function(){this.initChessboard(!0),this.$nextTick((function(){this.onPiecePlaceAudio=new Audio(i("489f"))}))},methods:{updateState:function(e){var t=this;this.validator.reset(),e.forEach((function(e){t.validator.move(e,{sloppy:!0})})),this.list_pieces=this.initChessboard(!1)},undo:function(){this.validator.undo()},_xPositionToFile:function(e){var t="h";return 1==e?t="a":2==e?t="b":3==e?t="c":4==e?t="d":5==e?t="e":6==e?t="f":7==e&&(t="g"),t},bCantMove:function(){return this.validator.in_draw()||this.validator.in_stalemate()||this.validator.in_threefold_repetition()||this.validator.in_checkmate()},got_tapped:function(e){if(e.preventDefault(),"cbrd"!=e.target.className){this.isTouchEvent=!0;var t=0;if(this.bCantMove())return;if(0==this.piece_selected){this.has_piece_selected=!0,t=e.target.className.split(" ")[1];var i=this.getPieceById(t);this.getValidMoves(i.position.x,i.position.y);var s=this.$refs[t][0];this.piece_selected=s}else{var o=document.body.getBoundingClientRect(),n=e.target.offsetParent.getBoundingClientRect(),a=n.top-o.top,r=n.left-o.left,c=e.target.offsetParent.clientWidth,l=0,h=0;l=e.changedTouches[0].pageY-a,h=e.changedTouches[0].pageX-r,t=this.piece_selected.id;var d=this.list_pieces.indexOf(this.getPieceById(t)),p=0,u=0;this.board_side?(p=8-(Math.ceil(800*l/c/100)-1),u=Math.ceil(800*h/c/100)):(p=Math.ceil(800*l/c/100),u=8-(Math.ceil(800*h/c/100)-1));var f=this._xPositionToFile(u),v=p,m=this._xPositionToFile(this.list_pieces[d].position.x),g=this.list_pieces[d].position.y,_=this.getValidMove({from:m+g,to:f+v});this.debug=_,this.makeMove(_,{posX:h,posY:l,new_positionX:u,new_positionY:p,from_file:m,from_rank:g,to_file:f,to_rank:v,indexInList:d,input:!0}),this.piece_selected=0,this.isTouchEvent=!1}}},got_selected:function(e){if("cbrd"!=e.target.className){if(this.bCantMove())return;this.has_piece_selected=!0;var t=e.target.className.split(" ")[1],i=this.getPieceById(t);this.getValidMoves(i.position.x,i.position.y);var s=this.$refs[t][0];this.piece_selected=s;var o=e.target.offsetParent.getBoundingClientRect();window.console.log("prova"),"mousedown"===e.type?(window.console.log("QUa"),this.piece_selected.got_selected(!1),this.piece_selected.set_mouse_pos(e.clientY-o.top-this.piece_selected.$el.clientHeight/2,e.clientX-o.left-this.piece_selected.$el.clientHeight/2)):(document.body.classList.toggle("stop-scroll"),U=e,this.piece_selected.set_mouse_pos(e.touches[0].clientY-o.top-this.piece_selected.$el.clientHeight,e.touches[0].clientX-o.left-this.piece_selected.$el.clientHeight/2))}},logListPieces:function(){},stopDrag:function(e){if(this.has_piece_selected){var t=e.target.offsetParent.getBoundingClientRect(),i=e.target.offsetParent.clientWidth,s=0,o=0;"mouseup"===e.type?(s=e.clientY-t.top,o=e.clientX-t.left):(document.body.classList.toggle("stop-scroll"),s=U.changedTouches[0].pageY-t.top,o=U.changedTouches[0].pageX-t.left,this.debug=s+" - "+o);var n=this.piece_selected.id,a=this.list_pieces.indexOf(this.getPieceById(n)),r=0,c=0;this.board_side?(r=8-(Math.ceil(800*s/i/100)-1),c=Math.ceil(800*o/i/100)):(r=Math.ceil(800*s/i/100),c=8-(Math.ceil(800*o/i/100)-1));var l=this._xPositionToFile(c),h=r,d=this._xPositionToFile(this.list_pieces[a].position.x),p=this.list_pieces[a].position.y,u=this.getValidMove({from:d+p,to:l+h});this.debug=u,this.piece_selected.got_unselected(),this.makeMove(u,{posX:o,posY:s,new_positionX:c,new_positionY:r,from_file:d,from_rank:p,to_file:l,to_rank:h,indexInList:a,input:!1})}},while_selected:function(e){if(0!=this.piece_selected){var t=e.target.offsetParent.getBoundingClientRect();"mousemove"===e.type?this.piece_selected.set_mouse_pos(e.y-t.top-this.piece_selected.$el.clientHeight/2,e.x-t.left-this.piece_selected.$el.clientHeight/2):(U=e,this.piece_selected.set_mouse_pos(e.touches[0].clientY-t.top-this.piece_selected.$el.clientHeight,e.touches[0].clientX-t.left-this.piece_selected.$el.clientHeight/2))}},getValidMove:function(e){var t=!1,i=0;while(!t&&i<this.valid_moves.length)e.from==this.valid_moves[i].from&&e.to==this.valid_moves[i].to?t=!0:i+=1;return this.valid_moves[i]},swapPieces:function(e){var t=void 0,i=void 0;if(e)if(this.turn){t=this.getPieceIDByColorAndType("w","k");var s=this.$refs[t][0];i=this.getPieceIDBySquare({x:8,y:1});var o=this.$refs[i][0],n=this.setPieceLocationInBoard(6,1),a=this.setPieceLocationInBoard(7,1);o.setPosition(n.y,n.x),s.setPosition(a.y,a.x);var r=this.getPiecePositionInListById(t),c=this.getPiecePositionInListById(i);this.list_pieces[r].position.x=7,this.list_pieces[r].position.y=1,this.list_pieces[c].position.x=6,this.list_pieces[c].position.y=1}else{t=this.getPieceIDByColorAndType("b","k"),s=this.$refs[t][0],i=this.getPieceIDBySquare({x:8,y:8}),o=this.$refs[i][0];var l=this.setPieceLocationInBoard(6,8),h=this.setPieceLocationInBoard(7,8);o.setPosition(l.y,l.x),s.setPosition(h.y,h.x);var d=this.getPiecePositionInListById(t),p=this.getPiecePositionInListById(i);this.list_pieces[d].position.x=7,this.list_pieces[d].position.y=8,this.list_pieces[p].position.x=6,this.list_pieces[p].position.y=8}else if(this.turn){t=this.getPieceIDByColorAndType("w","k"),s=this.$refs[t][0],i=this.getPieceIDBySquare({x:1,y:1}),o=this.$refs[i][0];var u=this.setPieceLocationInBoard(4,1),f=this.setPieceLocationInBoard(3,1);o.setPosition(u.y,u.x),s.setPosition(f.y,f.x);var v=this.getPiecePositionInListById(t),m=this.getPiecePositionInListById(i);this.list_pieces[v].position.x=3,this.list_pieces[v].position.y=1,this.list_pieces[m].position.x=4,this.list_pieces[m].position.y=1}else{t=this.getPieceIDByColorAndType("b","k"),s=this.$refs[t][0],i=this.getPieceIDBySquare({x:1,y:8}),o=this.$refs[i][0];var g=this.setPieceLocationInBoard(4,8),_=this.setPieceLocationInBoard(3,8);o.setPosition(g.y,g.x),s.setPosition(_.y,_.x);var b=this.getPiecePositionInListById(t),y=this.getPiecePositionInListById(i);this.list_pieces[b].position.x=3,this.list_pieces[b].position.y=8,this.list_pieces[y].position.x=4,this.list_pieces[y].position.y=8}},getPieceByPosition:function(e){var t=!1,i=0,s=void 0;while(!t&&i<this.list_pieces.length)this.list_pieces[i].position.x==e.x&&this.list_pieces[i].position.y==e.y?(s=this.list_pieces[i],t=!0):i+=1;return s},getPieceById:function(e){var t=void 0,i=!1,s=0;while(!i)parseInt(this.list_pieces[s].id)==e?(t=this.list_pieces[s],i=!0):s+=1;return t},getValidMoves:function(e,t){var i=this,s=this._xPositionToFile(e),o=s+t,n=this.validator.moves({verbose:!0,square:o});n.forEach((function(e){i.valid_moves.push({from:e.from,to:e.to,flags:e.flags})}))},getPiecePositionInListById:function(e){var t=!1,i=0;while(!t)parseInt(this.list_pieces[i].id)==e?t=!0:i+=1;return i},setPieceLocationInBoard:function(e,t){var i={x:0,y:0};return this.board_side?(i.x=100*e-100,i.y=100*(8-t)):(i.y=100*t-100,i.x=100*(8-e)),i},fileToNumber:function(e){var t=8;return t="a"==e?1:"b"==e?2:"c"==e?3:"d"==e?4:"e"==e?5:"f"==e?6:"g"==e?7:8,t},resetChessboard:function(){this.validator.in_checkmate()&&this.$refs.chessBoard.classList.toggle("gameover"),this.validator=new H,this.moves_history=[],this.board_side=!0,this.list_pieces=[],this.initChessboard(!0)},initChessboard:function(){var e=void 0;e=this.validator;for(var t=0,i=0;i<64;i++){var s=e.get(e.SQUARES[i]);if(null!=s){var o=e.SQUARES[i],n=parseInt(o[1]),a=o[0];a=this.fileToNumber(a);var r={id:t,type:s.type,color:s.color,position:{x:a,y:n}};this.list_pieces.push(r),t+=1}}},getPieceIDByColorAndType:function(e,t){var i=!1,s=0,o=void 0;while(!i&&s<this.list_pieces.length)this.list_pieces[s].color===e&&this.list_pieces[s].type===t?(i=!0,o=this.list_pieces[s].id):s+=1;return!!i&&o},toggleCheck:function(e){var t=e?"w":"b",i=this.getPieceIDByColorAndType(t,"k");this.$refs[i][0].$el.classList.toggle("check")},flipBoard:function(){var e=this;this.board_side=!this.board_side,this.list_pieces.forEach((function(t){var i=e.setPieceLocationInBoard(t.position.x,t.position.y);e.$refs[t.id][0].setPosition(i.y,i.x)}))},getPieceIndexBySquare:function(e){var t=!1,i=0;while(!t&&i<this.list_pieces.length)this.list_pieces[i].position.x==e.x&&this.list_pieces[i].position.y==e.y?t=!0:i+=1;return i},pushMove:function(e){this.moves_history.push(e)},getPieceIDBySquare:function(e){var t=!1,i=0,s=0;while(!t&&i<this.list_pieces.length)this.list_pieces[i].position.x==e.x&&this.list_pieces[i].position.y==e.y?(t=!0,s=this.list_pieces[i].id):i+=1;return s},makeMove:function(e){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:void 0,s="";if(void 0==t){e.length>4&&(s=e.length>5?"="+e[4]+e[5]:s="="+e[4],e=e.substring(0,e.length-1));var o=this.validator.move(e+s,{sloppy:!0});if(e=o.san,null==o)return;this.validator.undo()}if(void 0!=e&&null!=e){var n=this.$refs.chessBoard.clientHeight,a={posX:0,posY:0,new_positionX:0,new_positionY:0,from_file:0,from_rank:0,to_file:0,to_rank:0,indexInList:0};if(void 0==t){console.log(this.validator.moves()),e=this.validator.move(e);var r=this.fileToNumber(e.from[0]),c=e.from[1];a.new_positionX=this.fileToNumber(e.to[0]),a.new_positionY=e.to[1],a.from_file=e.from[0],a.from_rank=e.from[1],a.to_file=e.to[0],a.to_rank=e.to[1];var l=this.setPieceLocationInBoard(a.new_positionX,a.new_positionY);a.posX=l.x,a.posY=l.y,this.piece_selected=this.$refs[this.getPieceIDBySquare({x:r,y:parseInt(c)})][0],a.indexInList=this.getPieceIndexBySquare({x:r,y:c}),this.validator.undo()}else a={posX:t.posX,posY:t.posY,new_positionX:t.new_positionX,new_positionY:t.new_positionY,from_file:t.from_file,from_rank:t.from_rank,to_file:t.to_file,to_rank:t.to_rank,indexInList:t.indexInList},a.posY=100*Math.ceil(800*a.posY/n/100)-100,a.posX=100*Math.ceil(800*a.posX/n/100)-100;var h=0,d=!1;e.flags.includes("e")&&(h=this.turn?-1:1),0==h&&e.flags.includes("p")&&(d=!0);var p=d?{from:a.from_file+a.from_rank,to:a.to_file+a.to_rank,promotion:"q"}:{from:a.from_file+a.from_rank,to:a.to_file+a.to_rank},u=this.validator.move(p);if(e.flags.includes("k"))this.turn,this.swapPieces(!0);else if(e.flags.includes("q"))this.turn,this.swapPieces(!1);else{var f=this.getPieceByPosition({x:a.new_positionX,y:parseInt(a.new_positionY)+h});if(d&&(this.list_pieces[a.indexInList].type="q",this.piece_selected.className="w"),void 0!=f){var v=this.list_pieces.indexOf(f);this.list_pieces[a.indexInList].position.x=a.new_positionX,this.list_pieces[a.indexInList].position.y=a.new_positionY,a.indexInList>-1&&this.list_pieces.splice(v,1)}else this.list_pieces[a.indexInList].position.x=a.new_positionX,this.list_pieces[a.indexInList].position.y=a.new_positionY;this.debug=a.posX+"-"+a.posY,this.piece_selected.setPosition(a.posY,a.posX),this.debug="2"}if(this.pushMove({pieceType:this.list_pieces[a.indexInList].type,color:this.list_pieces[a.indexInList].color,from_file:a.from_file,from_rank:a.from_rank,to_file:a.to_file,to_rank:a.to_rank,captured:f,flags:e.flags,san:u.san}),this.debug="3",this.turn=!this.turn,this.onPiecePlaceAudio&&this.onPiecePlaceAudio.play(),this.debug="4",this.flipBoard(),this.validator.in_check()?(this.is_in_check&&this.toggleCheck(!this.turn),this.is_in_check=!0,this.toggleCheck(this.turn)):this.is_in_check&&(this.toggleCheck(!this.turn),this.is_in_check=!1),this.validator.in_checkmate()){var m=new Audio(i("9310"));m.play(),this.$refs.chessBoard.classList.toggle("gameover")}}if(this.has_piece_selected=!1,this.piece_selected=0,void 0!=t){var g=this.valid_moves.length;while(g>=0)this.valid_moves.pop(),g-=1}},debugMakeMove:function(){this.makeMove(this.debug[0]),this.debug.shift()}}},Q=K,J=(i("1960"),i("62ad")),W=i("0fd9"),Z=Object(l["a"])(Q,E,V,!1,null,"44014e68",null),ee=Z.exports;d()(Z,{VBtn:v["a"],VCol:J["a"],VRow:W["a"]});var te=function(){var e=this,t=e.$createElement,i=e._self._c||t;return i("h3",[e._v("Redirect")])},ie=[],se={name:"Signin",created:function(){}},oe=se,ne=Object(l["a"])(oe,te,ie,!1,null,null,null),ae=ne.exports,re=function(){var e=this,t=e.$createElement,i=e._self._c||t;return i("v-container",[[[i("v-content",[i("v-row",[i("Chessboard",{ref:"chessboard"})],1)],1)]]],2)},ce=[],le={name:"Game",components:{Chessboard:ee},data:function(){return{id:void 0}},methods:{changeID:function(e){this.id=e,this.listenForGameState()},sendEvent:function(){var e=Object(a["a"])(regeneratorRuntime.mark((function e(t){var s;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:if("move"!=t.type){e.next=5;break}return s=i("bc3a"),e.next=4,s({method:"post",url:"https://lichess.org/api/board/game/"+this.id+"/move/"+t.move,headers:{Authorization:"Bearer "+this.$store.getters.tokens,"Content-Type":"application/x-www-form-urlencoded"}}).catch((function(){return window.console.log("REDO"),null}));case 4:return e.abrupt("return",!0);case 5:case"end":return e.stop()}}),e,this)})));function t(t){return e.apply(this,arguments)}return t}(),gotEvent:function(e){var t=this;if(window.console.log(e),this.has_game_started){if(window.console.log(this.$refs["chessboard"]),"gameState"==e.type&&"started"==e.status){var i=e.moves.split(" "),s=i.length%2==0;window.console.log(s+" - "+this.game.is_player_white),(s&&this.game.is_player_white||!s&&!this.game.is_player_white)&&this.$refs["chessboard"].makeMove(i[i.length-1])}}else if("gameFull"==e.type&&(this.has_game_started=!0,window.console.log("Game started"),window.console.log(this.$parent.$parent.$parent.account.id),window.console.log(e.black.id),this.$parent.$parent.$parent.account.id==e.black.id&&(this.game.is_player_white=!1,this.$refs["chessboard"].is_player_white=!1,window.console.log("Is player white"),this.$refs["chessboard"].flipBoard()),""!=e.state.moves)){var o=e.state.moves.split(" ");o.forEach((function(e){t.$refs["chessboard"].makeMove(e)}))}},listenForGameState:function(){var e=Object(a["a"])(regeneratorRuntime.mark((function e(){var t,s=this;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:t=i("048a"),fetch("https://lichess.org/api/board/game/stream/"+this.id,{headers:{Authorization:"Bearer "+this.$store.getters.tokens}}).then((function(e){return window.console.log("response"),t(e.body)})).then((function(e){var t,i=e.getReader();i.read().then(t=function(e){e.done||(s.gotEvent(e.value),i.read().then(t))})})).catch((function(e){window.console.log("Errore: "+e)}));case 2:case"end":return e.stop()}}),e,this)})));function t(){return e.apply(this,arguments)}return t}()}},he=le,de=i("a523"),pe=Object(l["a"])(he,re,ce,!1,null,null,null),ue=pe.exports;d()(pe,{VContainer:de["a"],VContent:m["a"],VRow:W["a"]});var fe=function(){var e=this,t=e.$createElement,i=e._self._c||t;return i("v-container",[this.isSearchingForGame?[i("v-row",{staticStyle:{"padding-top":"10%"},attrs:{justify:"center",align:"center"}},[i("v-col",{staticClass:"d-flex justify-center",attrs:{cols:"12"}},[i("v-progress-circular",{attrs:{size:200,width:10,color:"primary",indeterminate:""}})],1),i("v-col",{staticClass:"d-flex justify-center"},[i("h2",[e._v("Looking for a match...")])])],1)]:[i("v-row",{attrs:{justify:"center"}},[i("v-col",{attrs:{cols:"10",sm:"10",md:"5"}},[i("v-row",[i("v-form",{ref:"form"},[i("v-row",{attrs:{align:"center",justify:"center"}},[i("v-col",{staticClass:"d-flex justify-center",attrs:{cols:"12"}},[i("h1",[e._v("Challenge a Human")])]),i("v-col",{staticClass:"d-flex justify-center",attrs:{cols:"12",sm:"6",md:"6"}},[i("v-btn-toggle",{attrs:{borderless:""},model:{value:e.seek.rated,callback:function(t){e.$set(e.seek,"rated",t)},expression:"seek.rated"}},[i("v-btn",{attrs:{value:!1}},[e._v("Unrated")]),i("v-btn",{attrs:{value:!0}},[e._v("Rated")])],1)],1),i("v-col",{attrs:{cols:"12",sm:"6",md:"6","order-md":"2"}},[i("v-select",{attrs:{items:["random","white","black"],label:"Color"},model:{value:e.seek.color,callback:function(t){e.$set(e.seek,"color",t)},expression:"seek.color"}})],1),i("v-col",{attrs:{sm:"12",md:"6",cols:"12"}},[i("v-subheader",[e._v("Time: "+e._s(e.seek.time)+" minutes")]),i("v-slider",{attrs:{"prepend-icon":"av_timer",min:0,max:15},model:{value:e.seek.time,callback:function(t){e.$set(e.seek,"time",t)},expression:"seek.time"}})],1),i("v-col",{attrs:{md:"6",cols:"12","order-md":"4"}},[i("v-subheader",[e._v("Time increment: "+e._s(e.getTime))]),i("v-slider",{attrs:{"prepend-icon":"alarm_add",min:0,max:180},model:{value:e.seek.increment,callback:function(t){e.$set(e.seek,"increment",t)},expression:"seek.increment"}})],1),i("v-col",{attrs:{md:"6",cols:"12","offset-md":"6","order-md":"5"}},[i("v-subheader",[e._v("Opponent's rating: "+e._s(e.seek.rating_range[0])+" - "+e._s(e.seek.rating_range[1]))]),i("v-range-slider",{attrs:{"prepend-icon":"score",max:2900,min:600},model:{value:e.seek.rating_range,callback:function(t){e.$set(e.seek,"rating_range",t)},expression:"seek.rating_range"}})],1),i("v-col",{staticClass:"d-flex justify-center",attrs:{md:"6",cols:"12","order-md":"6"}},[i("v-btn",{staticClass:"ma-2",attrs:{tile:"","x-large":"",color:"primary",dark:""},on:{click:function(t){return e.searchGame()}}},[e._v("Find Game")])],1)],1)],1)],1)],1),i("v-col",{attrs:{cols:"10",sm:"10",md:"5","offset-md":"1"}},[i("v-row",[i("v-form",{ref:"form"},[i("v-row",{attrs:{justify:"center"}},[i("v-col",{staticClass:"d-flex justify-center",attrs:{cols:"12"}},[i("h1",[e._v("Challenge Stockfish")])]),i("v-col",{staticClass:"d-flex justify-center",attrs:{cols:"12",md:"12"}},[i("v-btn-toggle",{attrs:{color:"primary","background-color":"secondary"},model:{value:e.seekAI.level,callback:function(t){e.$set(e.seekAI,"level",t)},expression:"seekAI.level"}},[i("v-btn",[e._v("1")]),i("v-btn",[e._v("2")]),i("v-btn",[e._v("3")]),i("v-btn",[e._v("4")]),i("v-btn",[e._v("5")]),i("v-btn",[e._v("6")]),i("v-btn",[e._v("7")]),i("v-btn",[e._v("8")])],1)],1),i("v-col",{attrs:{cols:"12",sm:"12",md:"12"}},[i("v-select",{attrs:{items:["random","white","black"],label:"Color"},model:{value:e.seekAI.color,callback:function(t){e.$set(e.seekAI,"color",t)},expression:"seekAI.color"}})],1),i("v-col",{staticClass:"d-flex justify-center",attrs:{md:"6",cols:"12","order-md":"6"}},[i("v-btn",{staticClass:"ma-2",attrs:{tile:"","x-large":"",color:"primary",dark:""},on:{click:function(t){return e.challengeStockfish()}}},[e._v("Challenge Stockfish")])],1)],1)],1)],1)],1)],1)]],2)},ve=[],me={name:"GameFinder",data:function(){return{seek:{rated:!1,time:0,increment:0,color:"random",rating_range:[1200,2e3]},seekAI:{level:0,color:"random"},isSearchingForGame:!1}},methods:{searchGame:function(){var e=Object(a["a"])(regeneratorRuntime.mark((function e(){return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:this.isSearchingForGame=!0;case 1:case"end":return e.stop()}}),e,this)})));function t(){return e.apply(this,arguments)}return t}(),challengeStockfish:function(){var e=Object(a["a"])(regeneratorRuntime.mark((function e(){var t,s,o=this;return regeneratorRuntime.wrap((function(e){while(1)switch(e.prev=e.next){case 0:return t=i("bc3a"),s=i("b383"),e.next=4,t({method:"post",url:"https://lichess.org/api/challenge/ai",data:s.stringify({level:this.seekAI.level+1,color:this.seekAI.color,ratingRange:this.seek.rating_range[0]+"-"+this.seek.rating_range[1]}),headers:{Authorization:"Bearer "+this.$store.getters.tokens,"Content-Type":"application/x-www-form-urlencoded"}}).catch((function(){o.isSearchingForGame=!1}));case 4:case"end":return e.stop()}}),e,this)})));function t(){return e.apply(this,arguments)}return t}()},computed:{getTime:function(){if(0==this.seek.increment)return"None";var e=Math.floor(this.seek.increment/60),t="";0!=e&&(t=e+" minutes");var i=this.seek.increment-60*e;return 0!=i&&(t=0!=e?t+" "+i+" seconds":i+" seconds"),t}}},ge=me,_e=i("a609"),be=i("4bd4"),ye=i("490a"),we=i("5963"),ke=i("b974"),xe=i("ba0d"),Ie=i("e0c7"),Pe=Object(l["a"])(ge,fe,ve,!1,null,null,null),$e=Pe.exports;d()(Pe,{VBtn:v["a"],VBtnToggle:_e["a"],VCol:J["a"],VContainer:de["a"],VForm:be["a"],VProgressCircular:ye["a"],VRangeSlider:we["a"],VRow:W["a"],VSelect:ke["a"],VSlider:xe["a"],VSubheader:Ie["a"]});var Le=i("ce5b"),Be=i.n(Le);i("bf40");s["default"].use(Be.a,{options:{customProperties:!0}});var Se={theme:{dark:!0,themes:{dark:{primary:"#333333",secondary:"#212121",accent:"#cddc39",error:"#f44336",warning:"#ff9800",info:"#e91e63",success:"#4caf50"},customProperties:!0}}},Ce=new Be.a(Se);s["default"].config.productionTip=!1,s["default"].use(B["a"]),s["default"].use(C["a"]),s["default"].use(O.a);var Te=new B["a"].Store({state:{token:null},mutations:{SET_TOKENS:function(e,t){e.token=t,window.console.log(e.token)},DELETE_TOKENS:function(e,t){e[t]=null}},getters:{tokens:function(e){return e.token}},actions:{LOGIN:function(e,t){e.commit("SET_TOKENS",t)},LOGOUT:function(e,t){e.commit("DELETE_TOKENS",t)}},plugins:[Object(S["a"])({getState:function(e){return window.$cookies.get(e)},setState:function(e,t){return window.$cookies.set(e,t)},removeState:function(e){return window.$cookies.remove(e)}})]}),Oe=new C["a"]({routes:[{path:"/",name:"homepage",component:ee},{path:"/board",name:"board",component:ee},{path:"/signin",name:"signin",component:ae},{path:"/game/:id?",name:"game",component:ue,props:!0},{path:"/find_game",name:"find_game",component:$e}]});new s["default"]({render:function(e){return e(L)},store:Te,router:Oe,vuetify:Ce}).$mount("#app")},"5c0b":function(e,t,i){"use strict";var s=i("7694"),o=i.n(s);o.a},"5fb3":function(e,t,i){},7694:function(e,t,i){},"797a":function(e,t,i){},9310:function(e,t,i){e.exports=i.p+"media/gameover.ce7eed20.mp3"},a532:function(e,t,i){"use strict";var s=i("797a"),o=i.n(s);o.a},d9ae:function(e,t,i){"use strict";var s=i("1be8"),o=i.n(s);o.a}});
//# sourceMappingURL=app.08aa64a0.js.map