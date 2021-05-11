from tkinter import *

class GUI_FileView:
		
	def __init__(self):
		self.root = Tk()
		
		self.objs = []
		self.camera = None
		self.lights = []
		
		self.TitleTxt = "Scene.py"
		
		self.bkg = "#080815"
		self.frm_bkg = "#202030"
		self.title_bkg = "#102545"
		self.hgl_col = "#a5a5a0"
		self.svBtn_bkg = self.title_bkg
		self.rndrBtn_bkg = "#00553f"		
		self.bdw = 4
	
		self.title_fnt = ("", 8,"bold","italic")
		self.heading_fnt = ("", 7, "italic")
		self.btn_fnt = ("", 7)
		self.txt_fnt = ("", 4)
		self.btn_fg = "#eee"
		self.txt_fg = "#a5a5a0"
		self.intrn_frmpadX = 30
		self.cont_frmpadY = 15
		

	def make(self):
		view_frm = Frame(self.root, bg=self.bkg)
		view_frm.pack(expand=1, fill=BOTH)		
		
		#title_bar
		title_frm = Frame(view_frm, bg=self.title_bkg)		
		title_lab =Label(title_frm,
									text=self.TitleTxt,
									bg=self.title_bkg,
									fg=self.txt_fg,
									font = self.title_fnt)
		title_lab.pack(side=LEFT, padx=20,pady=10)
		title_frm.pack(side=TOP,fill=X)
		
		#object_frame
		Ob_cont = Frame(view_frm, bg=self.bkg)
		Ob_cont.pack(expand=1,
								fill=BOTH,
								pady=self.cont_frmpadY)
		
		obj_lab =Label(Ob_cont,
									text= "Objects",
									bg=self.bkg,
									fg=self.txt_fg,
									font = self.heading_fnt,
									anchor=W)
		obj_lab.pack(fill=X)
							
		obj_frm = Frame(Ob_cont,
										highlightthickness=self.bdw,
										highlightbackground=self.hgl_col,
										bg=self.frm_bkg)
		obj_frm.pack(fill=BOTH,
								expand=1,
								padx=self.intrn_frmpadX)
		
		#camera_frame
		cam_cont = Frame(view_frm, bg=self.bkg)
		cam_cont.pack(fill=BOTH,
									expand=1,
									pady=self.cont_frmpadY)
		
		cam_lab =Label(cam_cont,
									text= "Camera",
									bg=self.bkg,
									fg=self.txt_fg,
									font = self.heading_fnt,
									anchor=W)
		cam_lab.pack(fill=X)
							
		cam_frm = Frame(cam_cont,
										highlightthickness=self.bdw,
										highlightbackground=self.hgl_col,
										bg=self.frm_bkg)			
		cam_frm.pack(fill=BOTH,
									expand=1,
									padx=self.intrn_frmpadX)
		
		#lights_frame
		Li_cont = Frame(view_frm, bg=self.bkg)
		Li_cont.pack(expand=1,
								fill=BOTH,
								pady=self.cont_frmpadY)
		
		lts_lab =Label(Li_cont,
									text="Lights",
									bg=self.bkg,
									fg=self.txt_fg,
									font = self.heading_fnt,
									anchor=W)
		lts_lab.pack(fill=X)
		
		lts_frm = Frame(Li_cont,
									highlightthickness=self.bdw,
									highlightbackground=self.hgl_col,
									bg=self.frm_bkg)		
		lts_frm.pack(fill=BOTH,
								expand=1,
								padx=self.intrn_frmpadX)
		
		#options_frame
		opts_frm = Frame(view_frm, bg=self.bkg)		
		save_btn = Button(opts_frm,
									text="Save",
									bg= self.svBtn_bkg,
									fg= self.btn_fg,
									width=5,
									font = self.btn_fnt,
									highlightthickness=self.bdw,
									highlightbackground= self.svBtn_bkg,
									relief = FLAT)
		save_btn.grid(row=0,column=0,padx=5,sticky=E)
		
		render_btn = Button(opts_frm,
									text="Render",
									bg=self.rndrBtn_bkg,
									fg= self.btn_fg,
									width=5,
									font = self.btn_fnt,
									highlightthickness = self.bdw,
									highlightbackground=self.rndrBtn_bkg,
									relief = FLAT)
		render_btn.grid(row=0,column=1,padx=5,sticky=W)
		
		opts_frm.columnconfigure(0, weight=1)
		opts_frm.columnconfigure(1, weight=1)		
		opts_frm.pack(pady=10,
									side=BOTTOM)

		
Win1 = GUI_FileView()
Win1.make()
Win1.root.mainloop()
